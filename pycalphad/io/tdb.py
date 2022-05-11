"""
The tdb module provides support for reading and writing databases in
Thermo-Calc TDB format.
"""

from pyparsing import CaselessKeyword, CharsNotIn, Group
from pyparsing import LineEnd, MatchFirst, OneOrMore, Optional, SkipTo
from pyparsing import ZeroOrMore, Suppress, White, Word, alphanums, alphas, nums
from pyparsing import delimitedList, ParseException
import re
from symengine.lib.symengine_wrapper import UniversalSet, Union, Complement
from symengine import sympify, And, Or, Not, EmptySet, Interval, Piecewise, Add, Mul, Pow
from symengine import Symbol, LessThan, StrictLessThan, S, E
from tinydb import where
from pycalphad import Database
from pycalphad.io.database import DatabaseExportError
from pycalphad.io.grammar import float_number, chemical_formula
from pycalphad.variables import Species
import pycalphad.variables as v
from pycalphad.io.tdb_keywords import expand_keyword, TDB_PARAM_TYPES
from pycalphad.core.utils import generate_symmetric_group
from collections import defaultdict, namedtuple
import ast
import sys
import inspect
import functools
import itertools
import getpass
import datetime
import warnings
import hashlib
from copy import deepcopy

# ast.Num is deprecated in Python 3.8 in favor of as ast.Constant
# Both are whitelisted for compatability across versions
_AST_WHITELIST = [ast.Add, ast.BinOp, ast.Call, ast.Constant, ast.Div,
                  ast.Expression, ast.Load, ast.Mult, ast.Name, ast.Num,
                  ast.Pow, ast.Sub, ast.UAdd, ast.UnaryOp, ast.USub]

def _sympify_string(math_string):
    "Convert math string into SymEngine object."
    # drop pound symbols ('#') since they denote function names
    # we detect those automatically
    expr_string = math_string.replace('#', '')
    # sympify doesn't recognize LN as ln()
    expr_string = \
        re.sub(r'(?<!\w)LN(?!\w)', 'ln', expr_string, flags=re.IGNORECASE)
    expr_string = \
        re.sub(r'(?<!\w)LOG(?!\w)', 'log', expr_string, flags=re.IGNORECASE)
    expr_string = \
        re.sub(r'(?<!\w)EXP(?!\w)', 'exp', expr_string,
               flags=re.IGNORECASE)

    # sympify uses eval, so we need to sanitize the input
    nodes = ast.parse(expr_string)
    nodes = ast.Expression(nodes.body[0].value)

    for node in ast.walk(nodes):
        if type(node) not in _AST_WHITELIST: #pylint: disable=W1504
            raise ValueError('Expression from TDB file not in whitelist: '
                             '{}'.format(expr_string))
    return sympify(expr_string).xreplace(v.supported_variables_in_databases).n()

def _parse_action(func):
    """
    Decorator for pyparsing parse actions to ease debugging.

    pyparsing uses trial & error to deduce the number of arguments a parse
    action accepts. Unfortunately any ``TypeError`` raised by a parse action
    confuses that mechanism.

    This decorator replaces the trial & error mechanism with one based on
    reflection. If the decorated function itself raises a ``TypeError`` then
    that exception is re-raised if the wrapper is called with less arguments
    than required. This makes sure that the actual ``TypeError`` bubbles up
    from the call to the parse action (instead of the one caused by pyparsing's
    trial & error).

    Modified slightly from the original for Py3 compatibility
    Source: Florian Brucker on StackOverflow
    http://stackoverflow.com/questions/10177276/pyparsing-setparseaction-function-is-getting-no-arguments
    """
    func_items = inspect.signature(func).parameters.items()
    func_args = [name for name, param in func_items
                 if param.kind == param.POSITIONAL_OR_KEYWORD]
    num_args = len(func_args)
    if num_args > 3:
        raise ValueError('Input function must take at most 3 parameters.')

    @functools.wraps(func)
    def action(*args):
        "Wrapped function."
        if len(args) < num_args:
            if action.exc_info:
                raise action.exc_info[0](action.exc_info[1], action.exc_info[2])
        action.exc_info = None
        try:
            return func(*args[:-(num_args + 1):-1])
        except TypeError as err:
            action.exc_info = sys.exc_info()
            raise err

    action.exc_info = None
    return action

@_parse_action
def _make_piecewise_ast(toks):
    """
    Convenience function for converting tokens into a piecewise symengine object.
    """
    cur_tok = 0
    expr_cond_pairs = []

    # Only one token: Not a piecewise function; just return the AST
    if len(toks) == 1:
        return _sympify_string(toks[0].strip(' ,'))

    while cur_tok < len(toks)-1:
        low_temp = toks[cur_tok]
        try:
            high_temp = toks[cur_tok+2]
        except IndexError:
            # No temperature limit specified
            high_temp = None

        if high_temp is None:
            expr_cond_pairs.append(
                (
                    _sympify_string(toks[cur_tok+1]),
                    And(low_temp <= v.T)
                )
            )
        else:
            expr_cond_pairs.append(
                (
                    _sympify_string(toks[cur_tok+1]),
                    And(low_temp <= v.T, v.T < high_temp)
                )
            )
        cur_tok = cur_tok + 2
    expr_cond_pairs.append((0, True))
    return Piecewise(*expr_cond_pairs)

class TCCommand(CaselessKeyword): #pylint: disable=R0903
    """
    Parser element for dealing with Thermo-Calc command abbreviations.
    """
    def parseImpl(self, instring, loc, doActions=True):
        # Find the end of the keyword by searching for an end character
        start = loc
        endchars = ' ():,'
        loc = -1
        for charx in endchars:
            locx = instring.find(charx, start)
            if locx != -1:
                # match the end-character closest to the start character
                if loc != -1:
                    loc = min(loc, locx)
                else:
                    loc = locx
        # if no end character found, just match the whole thing
        if loc == -1:
            loc = len(instring)
        try:
            res = expand_keyword([self.match], instring[start:loc])
            if len(res) > 1:
                self.errmsg = '{0!r} is ambiguous: matches {1}' \
                    .format(instring[start:loc], res)
                raise ParseException(instring, loc, self.errmsg, self)
            # res[0] is the unambiguous expanded keyword
            # in principle, res[0] == self.match
            return loc, res[0]
        except ValueError:
            pass
        raise ParseException(instring, loc, self.errmsg, self)

def _tdb_grammar(): #pylint: disable=R0914
    """
    Convenience function for getting the pyparsing grammar of a TDB file.
    """
    int_number = Word(nums).setParseAction(lambda t: [int(t[0])])
    # symbol name, e.g., phase name, function name
    symbol_name = Word(alphanums+'_:', min=1)
    ref_phase_name = symbol_name = Word(alphanums+'_-:()/', min=1)
    # species name, e.g., CO2, AL, FE3+
    species_name = Word(alphanums+'+-*/_.', min=1) + Optional(Suppress('%'))
    reference_key = Word(alphanums+':_-')('reference_key')
    # constituent arrays are colon-delimited
    # each subarray can be comma- or space-delimited
    constituent_array = Group(delimitedList(Group(OneOrMore(Optional(Suppress(',')) + species_name)), ':'))
    param_types = MatchFirst([TCCommand(param_type) for param_type in TDB_PARAM_TYPES])
    # Let symengine do heavy arithmetic / algebra parsing for us
    # a convenience function will handle the piecewise details
    func_expr = (float_number | ZeroOrMore(',').setParseAction(lambda t: 0.01)) + OneOrMore(SkipTo(';') \
        + Suppress(';') + ZeroOrMore(Suppress(',')) + Optional(float_number) + \
        Suppress(Optional(Word('Yy', exact=1))), stopOn=Word('Nn', exact=1)) + Suppress(Optional(Word('Nn', exact=1)))
    # ELEMENT
    cmd_element = TCCommand('ELEMENT') + Word(alphas+'/-', min=1, max=2) + ref_phase_name + \
        float_number + float_number + float_number + LineEnd()
    # SPECIES
    cmd_species = TCCommand('SPECIES') + species_name + chemical_formula + LineEnd()
    # TYPE_DEFINITION
    cmd_typedef = TCCommand('TYPE_DEFINITION') + \
        Suppress(White()) + CharsNotIn(' !', exact=1) + SkipTo(LineEnd())
    # FUNCTION
    cmd_function = TCCommand('FUNCTION') + symbol_name + \
        func_expr.setParseAction(_make_piecewise_ast) + \
        Optional(Suppress(reference_key)) + LineEnd()
    # ASSESSED_SYSTEMS
    cmd_ass_sys = TCCommand('ASSESSED_SYSTEMS') + SkipTo(LineEnd())
    # DEFINE_SYSTEM_DEFAULT
    cmd_defsysdef = TCCommand('DEFINE_SYSTEM_DEFAULT') + SkipTo(LineEnd())
    # DEFAULT_COMMAND
    cmd_defcmd = TCCommand('DEFAULT_COMMAND') + SkipTo(LineEnd())
    # DATABASE_INFO
    cmd_database_info = TCCommand('DATABASE_INFO') + SkipTo(LineEnd())
    # VERSION_DATE
    cmd_version_date = TCCommand('VERSION_DATE') + SkipTo(LineEnd())
    # REFERENCE_FILE
    cmd_reference_file = TCCommand('REFERENCE_FILE') + SkipTo(LineEnd())
    # ADD_REFERENCES
    cmd_add_ref = TCCommand('ADD_REFERENCES') + SkipTo(LineEnd())
    # LIST_OF_REFERENCES
    cmd_lor = TCCommand('LIST_OF_REFERENCES') + SkipTo(LineEnd())
    # TEMPERATURE_LIMITS
    cmd_templim = TCCommand('TEMPERATURE_LIMITS') + SkipTo(LineEnd())
    # PHASE
    cmd_phase = TCCommand('PHASE') + symbol_name + \
        Suppress(White()) + CharsNotIn(' !', min=1) + Suppress(White()) + \
        Suppress(int_number) + Group(OneOrMore(float_number)) + \
        Suppress(SkipTo(LineEnd()))
    # CONSTITUENT
    cmd_constituent = TCCommand('CONSTITUENT') + symbol_name + \
        Suppress(White()) + Suppress(':') + constituent_array + \
        Suppress(':') + LineEnd()
    # PARAMETER
    cmd_parameter = TCCommand('PARAMETER') + param_types + \
        Suppress('(') + symbol_name + \
        Optional(Suppress('&') + Word(alphas+'/-', min=1, max=2), default=None) + \
        Suppress(',') + constituent_array + \
        Optional(Suppress(';') + int_number, default=0) + \
        Suppress(')') + func_expr.setParseAction(_make_piecewise_ast) + \
        Optional(Suppress(reference_key)) + LineEnd()
    # ZEROVOLUME_SPECIES
    cmd_zerovolume = TCCommand('ZEROVOLUME_SPECIES') + SkipTo(LineEnd())
    # DIFFUSION
    cmd_diffusion = TCCommand('DIFFUSION') + SkipTo(LineEnd())
    # Now combine the grammar together
    all_commands = cmd_element | \
                    cmd_species | \
                    cmd_typedef | \
                    cmd_function | \
                    cmd_ass_sys | \
                    cmd_defsysdef | \
                    cmd_defcmd | \
                    cmd_database_info | \
                    cmd_version_date | \
                    cmd_reference_file | \
                    cmd_add_ref | \
                    cmd_lor | \
                    cmd_templim | \
                    cmd_phase | \
                    cmd_constituent | \
                    cmd_parameter | \
                    cmd_zerovolume | \
                    cmd_diffusion
    return all_commands

def _process_typedef(targetdb, typechar, line):
    """
    Process a TYPE_DEFINITION command.

    Assumes all phases are entered into the database already and that the
    database defines _typechar_map, which defines a map of typechar to the
    phases that use it. Any phases that in the typechar dict for this will have
    the model_hints updated based on this type definition, regardless of which
    phase names may be defined in this TYPE_DEF line.

    """
    matching_phases = targetdb._typechar_map[typechar]
    del targetdb._typechar_map[typechar]
    # GES A_P_D BCC_A2 MAGNETIC  -1    0.4
    tokens = line.replace(',', '').split()
    if len(tokens) < 4:
        return
    keyword = expand_keyword(['DISORDERED_PART', 'MAGNETIC'], tokens[3].upper())[0]
    if len(keyword) == 0:
        raise ValueError('Unknown type definition keyword: {}'.format(tokens[3]))
    if len(matching_phases) == 0:
        warnings.warn(f"The type definition character `{typechar}` in `TYPE_DEFINITION {typechar} {line}` is not used by any phase.")
    if keyword == 'MAGNETIC':
        # Magnetic model, both IHJ and Xiong models use these model hints when
        # constructing Model instances, despite being prefixed `ihj_magnetic_`
        model_hints = {
            'ihj_magnetic_afm_factor': float(tokens[4]),
            'ihj_magnetic_structure_factor': float(tokens[5])
        }
        for phase_name in matching_phases:
            targetdb.phases[phase_name].model_hints.update(model_hints)

    # GES A_P_D L12_FCC DIS_PART FCC_A1
    if keyword == 'DISORDERED_PART':
        # order-disorder model: since we need to add model_hints to both the
        # ordered and disorderd phase, we special case to update the phase
        # names defined by the TYPE_DEF, rather than the updating the phases
        # with matching typechars.
        ordered_phase = tokens[2].upper()
        disordered_phase = tokens[4].upper()
        hint = {
            'ordered_phase': ordered_phase,
            'disordered_phase': disordered_phase,
        }
        if ordered_phase in targetdb.phases:
            targetdb.phases[ordered_phase].model_hints.update(hint)
        else:
            raise ValueError(f"The {ordered_phase} phase is not in the database, but is defined by: `TYPE_DEFINTION {typechar} {line}`")
        if disordered_phase in targetdb.phases:
            targetdb.phases[disordered_phase].model_hints.update(hint)
        else:
            raise ValueError(f"The {disordered_phase} phase is not in the database, but is defined by: `TYPE_DEFINTION {typechar} {line}`")


phase_options = {'ionic_liquid_2SL': 'Y',
                 'symmetry_FCC_4SL': 'F',
                 'symmetry_BCC_4SL': 'B',
                 'liquid': 'L',
                 'gas': 'G',
                 'aqueous': 'A',
                 'charged_phase': 'I'}
inv_phase_options = dict([reversed(i) for i in phase_options.items()])


def _process_phase(targetdb, name, typedefs, subls):
    """
    Process the PHASE command.
    """
    splitname = name.split(':')
    phase_name = splitname[0].upper()
    options = ''
    if len(splitname) > 1:
        options = splitname[1]
    targetdb.add_structure_entry(phase_name, phase_name)
    model_hints = {}
    for option in inv_phase_options.keys():
        if option in options:
            model_hints[inv_phase_options[option]] = True

    for typedef_char in list(typedefs):
        targetdb._typechar_map[typedef_char].append(phase_name)

    # Model hints are updated later based on the type definitions
    targetdb.add_phase(phase_name, model_hints, subls)

def _process_parameter(targetdb, param_type, phase_name, diffusing_species,
                       constituent_array, param_order, param, ref=None):
    """
    Process the PARAMETER command.
    """
    # sorting lx is _required_ here: see issue #17 on GitHub
    targetdb.add_parameter(param_type, phase_name.upper(),
                           [[c.upper() for c in sorted(lx)]
                            for lx in constituent_array.asList()],
                           param_order, param, ref, diffusing_species, force_insert=False)

def _unimplemented(*args, **kwargs): #pylint: disable=W0613
    """
    Null function.
    """
    pass

def _process_species(db, sp_name, sp_comp, charge=0, *args):
    """Add a species to the Database. If charge not specified, the Species will be neutral."""
    # process the species composition list of [element1, ratio1, element2, ratio2, ..., elementN, ratioN]
    constituents = {sp_comp[i]: sp_comp[i+1] for i in range(0, len(sp_comp), 2)}
    db.species.add(Species(sp_name, constituents, charge=charge))

def _process_reference_state(db, el, refphase, mass, H298, S298):
    db.refstates[el] = {
        'phase': refphase,
        'mass': mass,
        'H298': H298,
        'S298': S298,
    }

def _setitem_raise_duplicates(dictionary, key, value):
    if key in dictionary:
        raise ValueError("TDB contains duplicate FUNCTION {}".format(key))
    dictionary[key] = value

_TDB_PROCESSOR = {
    'ELEMENT': lambda db, el, ref_phase, mass, h, s: (db.elements.add(el), _process_reference_state(db, el, ref_phase, mass, h, s), _process_species(db, el, [el, 1], 0)),
    'SPECIES': _process_species,
    'TYPE_DEFINITION': lambda db, typechar, line: db._typedefs_queue.append((typechar, line)),
    'FUNCTION': lambda db, name, sym: _setitem_raise_duplicates(db.symbols, name, sym),
    'DEFINE_SYSTEM_DEFAULT': _unimplemented,
    'ASSESSED_SYSTEMS': _unimplemented,
    'DEFAULT_COMMAND': _unimplemented,
    'DATABASE_INFO': _unimplemented,
    'VERSION_DATE': _unimplemented,
    'REFERENCE_FILE': _unimplemented,
    'ADD_REFERENCES': _unimplemented,
    'LIST_OF_REFERENCES': _unimplemented,
    'TEMPERATURE_LIMITS': _unimplemented,
    'PHASE': _process_phase,
    'CONSTITUENT': \
        lambda db, name, c: db.add_phase_constituents(
            name.split(':')[0].upper(), c),
    'PARAMETER': _process_parameter,
    'ZEROVOLUME_SPECIES': _unimplemented,
    'DIFFUSION': _unimplemented,
}

def to_interval(relational):
    if isinstance(relational, And):
        result = UniversalSet()
        for i in relational.args:
            result = result.intersection(to_interval(i))
        return result
    elif isinstance(relational, Or):
        return Union(*[to_interval(i) for i in relational.args])
    elif isinstance(relational, Not):
        return Complement(*[to_interval(i) for i in relational.args])
    if relational == S.true:
        return Interval(S.NegativeInfinity, S.Infinity, left_open=True, right_open=True)

    if len(relational.free_symbols) != 1:
        raise ValueError(f'Relational must only have one free symbol. Got {len(relational.free_symbols)} ({relational.free_symbols}) for relational {relational}')
    if len(relational.args) != 2:
        raise ValueError(f'Relational must only have two arguments. Got {len(relational.args)} ({relational.args}) for relational {relational}')
    free_symbol = list(relational.free_symbols)[0]
    lhs = relational.args[0]
    rhs = relational.args[1]
    if isinstance(relational, LessThan):
        if rhs == free_symbol:
            return Interval(lhs, S.Infinity, left_open=False, right_open=True)
        else:
            return Interval(S.NegativeInfinity, rhs, left_open=True, right_open=False)
    elif isinstance(relational, StrictLessThan):
        if rhs == free_symbol:
            return Interval(lhs, S.Infinity, left_open=True, right_open=False)
        else:
            return Interval(S.NegativeInfinity, rhs, left_open=False, right_open=True)
    else:
        raise ValueError('Unsupported Relational: {}'.format(relational.__class__.__name__))


class TCPrinter(object):
    """
    Prints Thermo-Calc style function expressions.
    """
    def doprint(self, expr):
        return self._print_Piecewise(expr)

    def _stringify_expr(self, expr):
        if isinstance(expr, Add):
            terms = self._stringify_expr(expr.args[0])
            for arg in expr.args[1:]:
                adding_term = self._stringify_expr(arg)
                if adding_term[0] == '-':
                    terms += adding_term
                else:
                    terms += ' + ' + adding_term
            return terms
        elif isinstance(expr, Mul):
            terms = self._stringify_expr(expr.args[0])
            for arg in expr.args[1:]:
                terms += ' * ' + self._stringify_expr(arg)
            return terms
        elif isinstance(expr, Pow):
            if expr.args[0] == E:
                # This is the exponential function
                terms = 'exp(' + self._stringify_expr(expr.args[1]) + ')'
            else:
                argument = self._stringify_expr(expr.args[0])
                if isinstance(expr.args[0], (Add, Mul)):
                    argument = '( ' + argument + ' )'
                terms = argument + '**' + '(' + self._stringify_expr(expr.args[1]) + ')'
            return terms
        else:
            return str(expr)

    def _print_Piecewise(self, expr):
        # Filter out default zeros since they are implicit in a TDB
        filtered_args = [(x, cond) for x, cond in zip(*[iter(expr.args)]*2) if not ((cond == S.true) and (x == S.Zero))]
        exprs = [self._stringify_expr(x) for x, cond in filtered_args]
        # Only a small subset of piecewise functions can be represented
        # Need to verify that each cond's highlim equals the next cond's lowlim
        # to_interval() is used because symengine does not implement as_set()
        intervals = [to_interval(cond) for x, cond in filtered_args]
        intersected_intervals = UniversalSet()
        for i in intervals:
            intersected_intervals = intersected_intervals.intersection(i)
        if (len(intervals) > 1) and (intersected_intervals != EmptySet):
            raise ValueError('Overlapping intervals cannot be represented: {}'.format(intervals))
        continuous_interval = Interval(intervals[0].args[0], intervals[-1].args[1], False, True)
        # XXX: Wait, should this be the intersection or the union of continuous_interval?
        if Union(*intervals).union(continuous_interval) != continuous_interval:
            raise ValueError('Piecewise intervals must be continuous')
        if not all([cond.free_symbols == {v.T} for x, cond in filtered_args]):
            raise ValueError('Only temperature-dependent piecewise conditions are supported')
        # Sort expressions based on intervals
        sortindices = [i[0] for i in sorted(enumerate(intervals), key=lambda x:x[1].args[0])]
        exprs = [exprs[idx] for idx in sortindices]
        # Infinity is implicit in TDB, so we shouldn't print it; ',' means use default value
        as_str = lambda x: ',' if (x == S.Infinity) or (x == S.NegativeInfinity) else str(x)
        if len(exprs) > 1:
            result = '{1} {0}; {2} Y'.format(exprs[0], as_str(intervals[0].args[0]),
                                             as_str(intervals[0].args[1]))
            result += 'Y'.join([' {0}; {1} '.format(expr,
                                                   as_str(i.args[1])) for i, expr in zip(intervals[1:], exprs[1:])])
            result += 'N'
        else:
            result = '{0} {1}; {2} N'.format(as_str(intervals[0].args[0]), exprs[0],
                                             as_str(intervals[0].args[1]))

        return result

def reflow_text(text, linewidth=80):
    """
    Add line breaks to ensure text doesn't exceed a certain line width.

    Parameters
    ----------
    text : str
    linewidth : int, optional

    Returns
    -------
    reflowed_text : str
    """
    lines = text.split("\n")
    linebreak_chars = [" ", "$"]
    output_lines = []
    for line in lines:
        if len(line) <= linewidth:
            output_lines.append(line)
        else:
            while len(line) > linewidth:
                linebreak_idx = linewidth - 1
                while linebreak_idx > 0 and line[linebreak_idx] not in linebreak_chars:
                    linebreak_idx -= 1
                # Need to check 2 (rather than zero) because we prepend newlines with 2 characters
                if linebreak_idx <= 2:
                    raise ValueError(f"Unable to reflow the following line of length {len(line)} below the maximum length of {linewidth}: \n{line}")
                output_lines.append(line[:linebreak_idx])
                if "$" in line:
                    # previous line was a comment
                    line = "$ " + line[linebreak_idx:]
                else:
                    # Always put some leading spaces at the start of a new line
                    # Otherwise TC may misunderstand the expression
                    line = "  " + line[linebreak_idx:]
            output_lines.append(line)
    return "\n".join(output_lines)


def _apply_new_symbol_names(dbf, symbol_name_map):
    """
    Push changes in symbol names through the SymEngine expressions in symbols and parameters

    Parameters
    ----------
    dbf : Database
        A pycalphad Database.
    symbol_name_map : dict
        Map of {old_symbol_name: new_symbol_name}
    """
    # first apply the rename to the keys
    dbf.symbols = {symbol_name_map.get(name, name): expr for name, expr in dbf.symbols.items()}
    # then propagate through to the symbol SymEngine expression values
    dbf.symbols = {name: S(expr).xreplace({Symbol(s): Symbol(v) for s, v in symbol_name_map.items()}) for name, expr in dbf.symbols.items()}
    # finally propagate through to the parameters
    for p in dbf._parameters.all():
        dbf._parameters.update({'parameter': S(p['parameter']).xreplace({Symbol(s): Symbol(v) for s, v in symbol_name_map.items()})}, doc_ids=[p.doc_id])


KNOWN_SUBLATTICE_SYMMETRY_RELATIONS = {
    # Keys should correspond to the model hints added via the `phase_options` dict
    "symmetry_FCC_4SL": [[0, 1, 2, 3]],
    "symmetry_BCC_4SL": [[0, 1], [2, 3]],
}


def add_phase_symmetry_ordering_parameters(dbf):
    for phase_name, phase_obj in dbf.phases.items():
        if phase_obj.model_hints.get("ordered_phase", "") == phase_name:
            for symmetry_hint, symmetry in KNOWN_SUBLATTICE_SYMMETRY_RELATIONS.items():
                if phase_obj.model_hints.get(symmetry_hint, False):
                    for param in dbf.search(where("phase_name") == phase_name):
                        const_array = param["constituent_array"]
                        for symm_unique_const_array in set(generate_symmetric_group(const_array, symmetry)) - {const_array}:
                            new_param = {key: val for key, val in param.items()}
                            new_param["constituent_array"] = symm_unique_const_array
                            new_param["_generated_by_symmetry_option"] = True  # flag to be able to remove it later and preserve the phase option
                            dbf._parameters.insert(new_param)


def _symmetry_added_parameter(dbf, param):
    """
    Return true if parameter belongs to a phase with an active symmetry
    option and the parameter was added by a symmetry option.
    """
    phase_obj = dbf.phases.get(param["phase_name"])
    if phase_obj is None:
        # Phase isn't in the database at all, so it's impossible for this parameter
        # to get added by symmetry
        return False
    for symm_hint in set(KNOWN_SUBLATTICE_SYMMETRY_RELATIONS.keys()).intersection(phase_obj.model_hints.keys()):
            if phase_obj.model_hints[symm_hint] and param.get("_generated_by_symmetry_option", False):
                return True
    return False


def write_tdb(dbf, fd, groupby='subsystem', if_incompatible='warn'):
    """
    Write a TDB file from a pycalphad Database object.

    The goal is to produce TDBs that conform to the most restrictive subset of database specifications. Some of these
    can be adjusted for automatically, such as the Thermo-Calc line length limit of 78. Others require changing the
    database in non-trivial ways, such as the maximum length of function names (8). The default is to warn the user when
    attempting to write an incompatible database and the user must choose whether to warn and write the file anyway or
    to fix the incompatibility.

    Currently the supported compatibility fixes are:
    - Line length <= 78 characters (Thermo-Calc)
    - Function names <= 8 characters (Thermo-Calc)

    The current unsupported fixes include:
    - Keyword length <= 2000 characters (Thermo-Calc)
    - Element names <= 2 characters (Thermo-Calc)
    - Phase names <= 24 characters (Thermo-Calc)

    Other TDB compatibility issues required by Thermo-Calc or other software should be reported to the issue tracker.

    Parameters
    ----------
    dbf : Database
        A pycalphad Database.
    fd : file-like
        File descriptor.
    groupby : ['subsystem', 'phase'], optional
        Desired grouping of parameters in the file.
    if_incompatible : string, optional ['raise', 'warn', 'fix']
        Strategy if the database does not conform to the most restrictive database specification.
        The 'warn' option (default) will write out the incompatible database with a warning.
        The 'raise' option will raise a DatabaseExportError.
        The 'ignore' option will write out the incompatible database silently.
        The 'fix' option will rectify the incompatibilities e.g. through name mangling.
    """
    # Before writing anything, check that the TDB is valid and take the appropriate action if not
    if if_incompatible not in ['warn', 'raise', 'ignore', 'fix']:
        raise ValueError('Incorrect options passed to \'if_invalid\'. Valid args are \'raise\', \'warn\', or \'fix\'.')
    # Handle function names > 8 characters
    long_function_names = {k for k in dbf.symbols.keys() if len(k) > 8}
    if len(long_function_names) > 0:
        if if_incompatible == 'raise':
            raise DatabaseExportError('The following function names are beyond the 8 character TDB limit: {}. Use the keyword argument \'if_incompatible\' to control this behavior.'.format(long_function_names))
        elif if_incompatible == 'fix':
            # if we are going to make changes, make the changes to a copy and leave the original object untouched
            dbf = deepcopy(dbf) # TODO: if we do multiple fixes, we should only copy once
            symbol_name_map = {}
            for name in long_function_names:
                hashed_name = 'F' + str(hashlib.md5(name.encode('UTF-8')).hexdigest()).upper()[:7] # this is implictly upper(), but it is explicit here
                symbol_name_map[name] = hashed_name
            _apply_new_symbol_names(dbf, symbol_name_map)
        elif if_incompatible == 'warn':
            warnings.warn('Ignoring that the following function names are beyond the 8 character TDB limit: {}. Use the keyword argument \'if_incompatible\' to control this behavior.'.format(long_function_names))

    # Begin constructing the written database
    writetime = datetime.datetime.now()
    maxlen = 78
    output = ""
    # Comment header block
    # Import here to prevent circular imports
    from pycalphad import __version__
    try:
        # getuser() will raise on Windows if it can't find a username: https://bugs.python.org/issue32731
        username = getpass.getuser()
    except:
        # if we can't find a good username, just choose a default and move on
        username = 'user'
    output += ("$" * maxlen) + "\n"
    output += "$ Date: {}\n".format(writetime.strftime("%Y-%m-%d %H:%M"))
    output += "$ Components: {}\n".format(', '.join(sorted(dbf.elements)))
    output += "$ Phases: {}\n".format(', '.join(sorted(dbf.phases.keys())))
    output += "$ Generated by {} (pycalphad {})\n".format(username, __version__)
    output += ("$" * maxlen) + "\n\n"
    for element in sorted(dbf.elements):
        ref = dbf.refstates.get(element, {})
        refphase = ref.get('phase', 'BLANK')
        mass = ref.get('mass', 0.0)
        H298 = ref.get('H298', 0.0)
        S298 = ref.get('S298', 0.0)
        output += "ELEMENT {0} {1} {2} {3} {4} !\n".format(element.upper(), refphase, mass, H298, S298)
    if len(dbf.elements) > 0:
        output += "\n"
    for species in sorted(dbf.species, key=lambda s: s.name):
        if species.name not in dbf.elements:
            # construct the charge part of the specie
            if species.charge != 0:
                if species.charge >0:
                    charge_sign = '+'
                else:
                    charge_sign = ''
                charge = '/{}{}'.format(charge_sign, species.charge)
            else:
                charge = ''
            species_constituents = ''.join(['{}{}'.format(el, val) for el, val in sorted(species.constituents.items(), key=lambda t: t[0])])
            output += "SPECIES {0} {1}{2} !\n".format(species.name.upper(), species_constituents, charge)
    if len(dbf.species) > 0:
        output += "\n"
    # Write FUNCTION block
    for name, expr in sorted(dbf.symbols.items()):
        if not isinstance(expr, Piecewise):
            # Non-piecewise exprs need to be wrapped to print
            # Otherwise TC's TDB parser will complain
            expr = Piecewise((expr, And(v.T >= 1, v.T < 10000)))
        expr = TCPrinter().doprint(expr).upper()
        if ';' not in expr:
            expr += '; N'
        output += "FUNCTION {0} {1} !\n".format(name.upper(), expr)
    output += "\n"
    # Boilerplate code
    output += "TYPE_DEFINITION % SEQ * !\n"
    output += "DEFINE_SYSTEM_DEFAULT ELEMENT 2 !\n"
    default_elements = [i.upper() for i in sorted(dbf.elements) if i.upper() == 'VA' or i.upper() == '/-']
    if len(default_elements) > 0:
        output += 'DEFAULT_COMMAND DEFINE_SYSTEM_ELEMENT {} !\n'.format(' '.join(default_elements))
    output += "\n"
    typedef_chars = list("^&*()'ABCDEFGHIJKLMNOPQSRTUVWXYZ")[::-1]
    #  Write necessary TYPE_DEF based on model hints
    typedefs = defaultdict(lambda: ["%"])
    for name, phase_obj in sorted(dbf.phases.items()):
        model_hints = phase_obj.model_hints.copy()
        possible_options = set(phase_options.keys()).intersection(model_hints)
        # Phase options are handled later
        for option in possible_options:
            del model_hints[option]
        if ('ordered_phase' in model_hints.keys()) and (model_hints['ordered_phase'] == name):
            new_char = typedef_chars.pop()
            typedefs[name].append(new_char)
            typedefs[model_hints['disordered_phase']].append(new_char)
            output += 'TYPE_DEFINITION {} GES AMEND_PHASE_DESCRIPTION {} DISORDERED_PART {} !\n'\
                .format(new_char, model_hints['ordered_phase'].upper(),
                        model_hints['disordered_phase'].upper())
            del model_hints['ordered_phase']
            del model_hints['disordered_phase']
        if ('disordered_phase' in model_hints.keys()) and (model_hints['disordered_phase'] == name):
            # We handle adding the correct typedef when we write the ordered phase
            del model_hints['ordered_phase']
            del model_hints['disordered_phase']
        if 'ihj_magnetic_afm_factor' in model_hints.keys():
            new_char = typedef_chars.pop()
            typedefs[name].append(new_char)
            output += 'TYPE_DEFINITION {} GES AMEND_PHASE_DESCRIPTION {} MAGNETIC {} {} !\n'\
                .format(new_char, name.upper(), model_hints['ihj_magnetic_afm_factor'],
                        model_hints['ihj_magnetic_structure_factor'])
            del model_hints['ihj_magnetic_afm_factor']
            del model_hints['ihj_magnetic_structure_factor']
        if len(model_hints) > 0:
            # Some model hints were not properly consumed
            raise ValueError('Not all model hints are supported: {}'.format(model_hints))
    # Perform a second loop now that all typedefs / model hints are consistent
    for name, phase_obj in sorted(dbf.phases.items()):
        # model_hints may also contain "phase options", e.g., ionic liquid
        model_hints = phase_obj.model_hints.copy()
        name_with_options = str(name.upper())
        possible_options = set(phase_options.keys()).intersection(model_hints.keys())
        if len(possible_options) > 0:
            name_with_options += ':'
        for option in possible_options:
            name_with_options += phase_options[option]
        output += "PHASE {0} {1}  {2} {3} !\n".format(name_with_options, ''.join(typedefs[name]),
                                                      len(phase_obj.sublattices),
                                                      ' '.join([str(i) for i in phase_obj.sublattices]))
        constituents = ':'.join([', '.join([spec.name for spec in sorted(subl)]) for subl in phase_obj.constituents])
        output += "CONSTITUENT {0} :{1}: !\n".format(name_with_options, constituents)
        output += "\n"

    # PARAMETERs by subsystem
    param_sorted = defaultdict(lambda: list())
    paramtuple = namedtuple('ParamTuple', ['phase_name', 'parameter_type', 'complexity', 'constituent_array',
                                           'parameter_order', 'diffusing_species', 'parameter', 'reference'])
    for param in dbf._parameters.all():
        if _symmetry_added_parameter(dbf, param):
            continue  # skip this parameter
        if groupby == 'subsystem':
            components = set()
            for subl in param['constituent_array']:
                components |= set(subl)
            if param['diffusing_species'] != Species(None):
                components |= {param['diffusing_species']}
            # Wildcard operator is not a component
            components -= {'*'}
            desired_active_pure_elements = [list(x.constituents.keys()) for x in components]
            components = set([el.upper() for constituents in desired_active_pure_elements for el in constituents])
            # Remove vacancy if it's not the only component (pure vacancy endmember)
            if len(components) > 1:
                components -= {'VA'}
            components = tuple(sorted([c.upper() for c in components]))
            grouping = components
        elif groupby == 'phase':
            grouping = param['phase_name'].upper()
        else:
            raise ValueError('Unknown groupby attribute \'{}\''.format(groupby))
        # We use the complexity parameter to help with sorting the parameters logically
        param_sorted[grouping].append(paramtuple(param['phase_name'], param['parameter_type'],
                                                 sum([len(i) for i in param['constituent_array']]),
                                                 param['constituent_array'], param['parameter_order'],
                                                 param['diffusing_species'], param['parameter'],
                                                 param['reference']))

    def write_parameter(param_to_write):
        constituents = ':'.join([','.join(sorted([i.name.upper() for i in subl]))
                         for subl in param_to_write.constituent_array])
        # TODO: Handle references
        paramx = param_to_write.parameter
        if not isinstance(paramx, Piecewise):
            # Non-piecewise parameters need to be wrapped to print correctly
            # Otherwise TC's TDB parser will fail
            paramx = Piecewise((paramx, And(v.T >= 1, v.T < 10000)))
        exprx = TCPrinter().doprint(paramx).upper()
        if ';' not in exprx:
            exprx += '; N'
        if param_to_write.diffusing_species != Species(None):
            ds = "&" + param_to_write.diffusing_species.name
        else:
            ds = ""
        return "PARAMETER {}({}{},{};{}) {} !\n".format(param_to_write.parameter_type.upper(),
                                                        param_to_write.phase_name.upper(),
                                                        ds,
                                                        constituents,
                                                        param_to_write.parameter_order,
                                                        exprx)
    if groupby == 'subsystem':
        for num_species in range(1, 5):
            subsystems = list(itertools.combinations(sorted([i.name.upper() for i in dbf.species]), num_species))
            for subsystem in subsystems:
                parameters = sorted(param_sorted[subsystem])
                if len(parameters) > 0:
                    output += "\n\n"
                    output += "$" * maxlen + "\n"
                    output += "$ {}".format('-'.join(sorted(subsystem)).center(maxlen, " ")[2:-1]) + "$\n"
                    output += "$" * maxlen + "\n"
                    output += "\n"
                    for parameter in parameters:
                        output += write_parameter(parameter)
        # Don't generate combinatorics for multi-component subsystems or we'll run out of memory
        if len(dbf.species) > 4:
            subsystems = [k for k in param_sorted.keys() if len(k) > 4]
            for subsystem in subsystems:
                parameters = sorted(param_sorted[subsystem])
                for parameter in parameters:
                    output += write_parameter(parameter)
    elif groupby == 'phase':
        for phase_name in sorted(dbf.phases.keys()):
            parameters = sorted(param_sorted[phase_name])
            if len(parameters) > 0:
                output += "\n\n"
                output += "$" * maxlen + "\n"
                output += "$ {}".format(phase_name.upper().center(maxlen, " ")[2:-1]) + "$\n"
                output += "$" * maxlen + "\n"
                output += "\n"
                for parameter in parameters:
                    output += write_parameter(parameter)
    else:
        raise ValueError('Unknown groupby attribute {}'.format(groupby))
    # Reflow text to respect character limit per line
    fd.write(reflow_text(output, linewidth=maxlen))


def read_tdb(dbf, fd):
    """
    Parse a TDB file into a pycalphad Database object.

    Parameters
    ----------
    dbf : Database
        A pycalphad Database.
    fd : file-like
        File descriptor.
    """
    lines = fd.read().upper()
    lines = lines.replace('\t', ' ')
    lines = lines.strip()
    # Split the string by newlines
    splitlines = lines.split('\n')
    # Remove extra whitespace inside line
    splitlines = [' '.join(k.split()) for k in splitlines]
    # Remove comments
    splitlines = [k.strip().split('$', 1)[0] for k in splitlines]
    # Remove everything after command delimiter, but keep the delimiter so we can split later
    splitlines = [k.split('!')[0] + ('!' if len(k.split('!')) > 1 else '') for k in splitlines]
    # Combine everything back together
    lines = ' '.join(splitlines)
    # Now split by the command delimeter
    commands = lines.split('!')

    # Temporarily track which typedef characters were used by which phase
    # before we process the type definitions
    # Map {typedef character: [phases using that typedef]}
    dbf._typechar_map = defaultdict(list)
    dbf._typedefs_queue = []  # queue of type defintion lines to process

    grammar = _tdb_grammar()

    for command in commands:
        if len(command) == 0:
            continue
        tokens = None
        try:
            tokens = grammar.parseString(command)
            _TDB_PROCESSOR[tokens[0]](dbf, *tokens[1:])
        except:
            print("Failed while parsing: " + command)
            print("Tokens: " + str(tokens))
            raise

    # Process type definitions last, updating model_hints for defined phases.
    for typechar, line in dbf._typedefs_queue:
        _process_typedef(dbf, typechar, line)
    # Raise warnings if there are any remaining type characters that one or more
    # phases expected to be defined
    for typechar, phases_expecting_typechar in dbf._typechar_map.items():
        warnings.warn(f"The type definition character `{typechar}` was defined in the following phases: "
                      f"{phases_expecting_typechar}, but no corresponding TYPE_DEFINITION line was found in the TDB.")
    del dbf._typechar_map
    del dbf._typedefs_queue

    dbf.process_parameter_queue()

    # Add phase option B/F parameters
    # Must occur after adding model hints and parameters
    add_phase_symmetry_ordering_parameters(dbf)


Database.register_format("tdb", read=read_tdb, write=write_tdb)
