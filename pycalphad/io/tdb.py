"""The tdb module provides support for reading and writing databases in
Thermo-Calc TDB format.
"""

from pyparsing import CaselessKeyword, CharsNotIn, Group
from pyparsing import LineEnd, MatchFirst, OneOrMore, Optional, Regex, SkipTo
from pyparsing import ZeroOrMore, Suppress, White, Word, alphanums, alphas, nums
from pyparsing import delimitedList, ParseException
import re
from sympy import sympify, And, Intersection, Union, EmptySet, Interval, Piecewise
from sympy import Symbol
from sympy.printing.str import StrPrinter
from pycalphad import Database
import pycalphad.variables as v
from pycalphad.io.tdb_keywords import expand_keyword, TDB_PARAM_TYPES
from collections import defaultdict, namedtuple
import ast
import sys
import inspect
import functools
import itertools
import getpass
import datetime

_AST_WHITELIST = [ast.Add, ast.BinOp, ast.Call, ast.Div, ast.Expression,
                  ast.Load, ast.Mult, ast.Name, ast.Num, ast.Pow, ast.Sub,
                  ast.UAdd, ast.UnaryOp, ast.USub]

def _sympify_string(math_string):
    "Convert math string into SymPy object."
    # drop pound symbols ('#') since they denote function names
    # we detect those automatically
    expr_string = math_string.replace('#', '')
    # sympify doesn't recognize LN as ln()
    expr_string = \
        re.sub(r'(?<!\w)LN(?!\w)', 'ln', expr_string, flags=re.IGNORECASE)
    expr_string = \
        re.sub(r'(?<!\w)EXP(?!\w)', 'exp', expr_string,
               flags=re.IGNORECASE)
    # Convert raw variables into StateVariable objects
    variable_fixes = {
        Symbol('T'): v.T,
        Symbol('P'): v.P,
        Symbol('R'): v.R
    }
    # sympify uses eval, so we need to sanitize the input
    nodes = ast.parse(expr_string)
    nodes = ast.Expression(nodes.body[0].value)

    for node in ast.walk(nodes):
        if type(node) not in _AST_WHITELIST: #pylint: disable=W1504
            raise ValueError('Expression from TDB file not in whitelist: '
                             '{}'.format(expr_string))
    return sympify(expr_string).xreplace(variable_fixes)

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
    num_args = len(inspect.getargspec(func).args)
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
    Convenience function for converting tokens into a piecewise sympy AST.
    """
    cur_tok = 0
    expr_cond_pairs = []

    # Only one token: Not a piecewise function; just return the AST
    if len(toks) == 1:
        return _sympify_string(toks[0])

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
    #expr_cond_pairs.append((0., True))
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
    # matching float w/ regex is ugly but is recommended by pyparsing
    float_number = Regex(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?') \
        .setParseAction(lambda t: [float(t[0])])
    # symbol name, e.g., phase name, function name
    symbol_name = Word(alphanums+'_:', min=1)
    # species name, e.g., CO2, AL, FE3+
    species_name = Word(alphanums+'+-*', min=1) + Optional(Suppress('%'))
    # constituent arrays are semicolon-delimited
    # each subarray can be comma- or space-delimited
    constituent_array = Group(
        delimitedList(Group(delimitedList(species_name, ',') & \
                            ZeroOrMore(species_name)
                           ), ':')
        )
    param_types = MatchFirst([TCCommand(param_type) for param_type in TDB_PARAM_TYPES])
    # Let sympy do heavy arithmetic / algebra parsing for us
    # a convenience function will handle the piecewise details
    func_expr = Optional(float_number) + OneOrMore(SkipTo(';') \
        + Suppress(';') + ZeroOrMore(Suppress(',')) + Optional(float_number) + \
        Suppress(Word('YNyn', exact=1)))
    # ELEMENT
    cmd_element = TCCommand('ELEMENT') + Word(alphas+'/-', min=1, max=2)
    # TYPE_DEFINITION
    cmd_typedef = TCCommand('TYPE_DEFINITION') + \
        Suppress(White()) + CharsNotIn(' !', exact=1) + SkipTo(LineEnd())
    # FUNCTION
    cmd_function = TCCommand('FUNCTION') + symbol_name + \
        func_expr.setParseAction(_make_piecewise_ast)
    # ASSESSED_SYSTEMS
    cmd_ass_sys = TCCommand('ASSESSED_SYSTEMS') + SkipTo(LineEnd())
    # DEFINE_SYSTEM_DEFAULT
    cmd_defsysdef = TCCommand('DEFINE_SYSTEM_DEFAULT') + SkipTo(LineEnd())
    # DEFAULT_COMMAND
    cmd_defcmd = TCCommand('DEFAULT_COMMAND') + SkipTo(LineEnd())
    # LIST_OF_REFERENCES
    cmd_lor = TCCommand('LIST_OF_REFERENCES') + SkipTo(LineEnd())
    # PHASE
    cmd_phase = TCCommand('PHASE') + symbol_name + \
        Suppress(White()) + CharsNotIn(' !', min=1) + Suppress(White()) + \
        Suppress(int_number) + Group(OneOrMore(float_number)) + LineEnd()
    # CONSTITUENT
    cmd_constituent = TCCommand('CONSTITUENT') + symbol_name + \
        Suppress(White()) + Suppress(':') + constituent_array + \
        Suppress(':') + LineEnd()
    # PARAMETER
    cmd_parameter = TCCommand('PARAMETER') + param_types + \
        Suppress('(') + symbol_name + Suppress(',') + constituent_array + \
        Optional(Suppress(';') + int_number, default=0) + Suppress(')') + \
        func_expr.setParseAction(_make_piecewise_ast)
    # Now combine the grammar together
    all_commands = cmd_element | \
                    cmd_typedef | \
                    cmd_function | \
                    cmd_ass_sys | \
                    cmd_defsysdef | \
                    cmd_defcmd | \
                    cmd_lor | \
                    cmd_phase | \
                    cmd_constituent | \
                    cmd_parameter
    return all_commands

def _process_typedef(targetdb, typechar, line):
    """
    Process the TYPE_DEFINITION command.
    """
    # GES A_P_D BCC_A2 MAGNETIC  -1    0.4
    tokens = line.replace(',', '').split()
    if len(tokens) < 4:
        return
    keyword = expand_keyword(['DISORDERED_PART', 'MAGNETIC'], tokens[3].upper())[0]
    if len(keyword) == 0:
        raise ValueError('Unknown keyword: {}'.format(tokens[3]))
    if keyword == 'MAGNETIC':
        # magnetic model (IHJ model assumed by default)
        targetdb.tdbtypedefs[typechar] = {
            'ihj_magnetic':[float(tokens[4]), float(tokens[5])]
        }
    # GES A_P_D L12_FCC DIS_PART FCC_A1
    if keyword == 'DISORDERED_PART':
        # order-disorder model
        targetdb.tdbtypedefs[typechar] = {
            'disordered_phase': tokens[4].upper(),
            'ordered_phase': tokens[2].upper()
        }
        if tokens[2].upper() in targetdb.phases:
            # Since TDB files do not enforce any kind of ordering
            # on the specification of ordered and disordered phases,
            # we need to handle the case of when either phase is specified
            # first. In this case, we imagine the ordered phase is
            # specified first. If the disordered phase is specified
            # first, we will have to catch it in _process_phase().
            targetdb.phases[tokens[2].upper()].model_hints.update(
                targetdb.tdbtypedefs[typechar]
            )

def _process_phase(targetdb, name, typedefs, subls):
    """
    Process the PHASE command.
    """
    splitname = name.split(':')
    phase_name = splitname[0].upper()
    options = None
    if len(splitname) > 1:
        options = splitname[1]
    targetdb.add_structure_entry(phase_name, phase_name)
    model_hints = {}
    for typedef in list(typedefs):
        if typedef in targetdb.tdbtypedefs.keys():
            if 'ihj_magnetic' in targetdb.tdbtypedefs[typedef].keys():
                model_hints['ihj_magnetic_afm_factor'] = \
                    targetdb.tdbtypedefs[typedef]['ihj_magnetic'][0]
                model_hints['ihj_magnetic_structure_factor'] = \
                    targetdb.tdbtypedefs[typedef]['ihj_magnetic'][1]
            if 'ordered_phase' in targetdb.tdbtypedefs[typedef].keys():
                model_hints['ordered_phase'] = \
                    targetdb.tdbtypedefs[typedef]['ordered_phase']
                model_hints['disordered_phase'] = \
                    targetdb.tdbtypedefs[typedef]['disordered_phase']
    targetdb.add_phase(phase_name, model_hints, subls)

def _process_parameter(targetdb, param_type, phase_name, #pylint: disable=R0913
                       constituent_array, param_order, param, ref=None):
    """
    Process the PARAMETER command.
    """
    # sorting lx is _required_ here: see issue #17 on GitHub
    targetdb.add_parameter(param_type, phase_name.upper(),
                           [[c.upper() for c in sorted(lx)]
                            for lx in constituent_array.asList()],
                           param_order, param, ref)

def _unimplemented(*args, **kwargs): #pylint: disable=W0613
    """
    Null function.
    """
    pass

_TDB_PROCESSOR = {
    'ELEMENT': lambda db, el: db.elements.add(el),
    'TYPE_DEFINITION': _process_typedef,
    'FUNCTION': lambda db, name, sym: db.symbols.__setitem__(name, sym),
    'DEFINE_SYSTEM_DEFAULT': _unimplemented,
    'ASSESSED_SYSTEMS': _unimplemented,
    'DEFAULT_COMMAND': _unimplemented,
    'LIST_OF_REFERENCES': _unimplemented,
    'PHASE': _process_phase,
    'CONSTITUENT': \
        lambda db, name, c: db.add_phase_constituents(
            name.split(':')[0].upper(), c),
    'PARAMETER': _process_parameter
}

class TCPrinter(StrPrinter):
    """
    Prints Thermo-Calc style function expressions.
    """
    def _print_Piecewise(self, expr):
        exprs = [self._print(arg.expr) for arg in expr.args]
        # Only a small subset of piecewise functions can be represented
        # Need to verify that each cond's highlim equals the next cond's lowlim
        intervals = [i.cond.as_set() for i in expr.args]
        if (len(intervals) > 1) and Intersection(intervals) != EmptySet():
            raise ValueError('Overlapping intervals cannot be represented: {}'.format(intervals))
        if not isinstance(Union(intervals), Interval):
            raise ValueError('Piecewise intervals must be continuous')
        if not all([arg.cond.free_symbols == {v.T} for arg in expr.args]):
            raise ValueError('Only temperature-dependent piecewise conditions are supported')
        # Sort expressions based on intervals
        sortindices = [i[0] for i in sorted(enumerate(intervals), key=lambda x:x[1].start)]
        exprs = [exprs[idx] for idx in sortindices]
        intervals = [intervals[idx] for idx in sortindices]

        if len(exprs) > 1:
            result = '{1} {0}; {2} Y'.format(exprs[0], self._print(intervals[0].start),
                                             self._print(intervals[0].end))
            result += 'Y'.join([' {0}; {1} '.format(expr,
                                                   self._print(i.end)) for i, expr in zip(intervals[1:], exprs[1:])])
            result += 'N'
        else:
            result = '{0} {1}; {2} N'.format(self._print(intervals[0].start), exprs[0],
                                             self._print(intervals[0].end))

        return result

    def _print_Function(self, expr):
        func_translations = {'log': 'ln', 'exp': 'exp'}
        if expr.func.__name__ in func_translations:
            return func_translations[expr.func.__name__] + "(%s)" % self.stringify(expr.args, ", ")
        else:
            raise TypeError("Unable to represent function: %s" %
                             expr.func.__name__)

    def blacklisted(self, expr):
        raise TypeError("Unable to represent expression: %s" %
                        expr.__class__.__name__)


    # blacklist all Matrix printing
    _print_SparseMatrix = \
    _print_MutableSparseMatrix = \
    _print_ImmutableSparseMatrix = \
    _print_Matrix = \
    _print_DenseMatrix = \
    _print_MutableDenseMatrix = \
    _print_ImmutableMatrix = \
    _print_ImmutableDenseMatrix = \
    blacklisted
    # blacklist other operations
    _print_Derivative = \
    _print_Integral = \
    blacklisted
    # blacklist some logical operations
    # These should never show up outside a piecewise function
    # Piecewise handles them directly
    _print_And = \
    _print_Or = \
    _print_Not = \
    blacklisted
    # blacklist some python expressions
    _print_list = \
    _print_tuple = \
    _print_Tuple = \
    _print_dict = \
    _print_Dict = \
    blacklisted


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
    ""
    lines = text.split("\n")
    linebreak_chars = [" "]
    output_lines = []
    for line in lines:
        if len(line) <= linewidth:
            output_lines.append(line)
        else:
            while len(line) > linewidth:
                linebreak_idx = linewidth-1
                while line[linebreak_idx] not in linebreak_chars:
                    linebreak_idx -= 1
                output_lines.append(line[:linebreak_idx])
                line = "  " + line[linebreak_idx:]
            output_lines.append(line)
    return "\n".join(output_lines)


def write_tdb(dbf, fd):
    """
    Write a TDB file from a pycalphad Database object.

    Parameters
    ----------
    dbf : Database
        A pycalphad Database.
    fd : file-like
        File descriptor.
    """
    writetime = datetime.datetime.now()
    output = ""
    # Comment header block
    # Import here to prevent circular imports
    from pycalphad import __version__
    output += ("$" * 80) + "\n"
    output += "$ Date: {}\n".format(writetime.strftime("%Y-%m-%d %H:%M"))
    output += "$ Components: {}\n".format(', '.join(sorted(dbf.elements)))
    output += "$ Phases: {}\n".format(', '.join(sorted(dbf.phases.keys())))
    output += "$ Generated by {} (pycalphad {})\n".format(getpass.getuser(), __version__)
    output += ("$" * 80) + "\n\n"
    for element in sorted(dbf.elements):
        output += "ELEMENT {0} BLANK 0 0 0 !\n".format(element.upper())
    if len(dbf.elements) > 0:
        output += "\n"
    for species in sorted(dbf.species):
        output += "SPECIES {0} !\n".format(species.upper())
    if len(dbf.species) > 0:
        output += "\n"
    # Write FUNCTION block
    for name, expr in sorted(dbf.symbols.items()):
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
    for name, phase_obj in sorted(dbf.phases.items()):
        #  Write necessary TYPE_DEF based on model hints
        typedefs = ["%"]
        model_hints = phase_obj.model_hints.copy()
        if ('ordered_phase' in model_hints.keys()) and (model_hints['ordered_phase'] == name):
            new_char = typedef_chars.pop()
            typedefs.append(new_char)
            output += 'TYPE_DEFINITION {} GES AMEND_PHASE_DESCRIPTION {} DISORDERED_PART {} !\n'\
                .format(new_char, model_hints['ordered_phase'].upper(),
                        model_hints['disordered_phase'].upper())
            del model_hints['ordered_phase']
            del model_hints['disordered_phase']
        if 'ihj_magnetic_afm_factor' in model_hints.keys():
            new_char = typedef_chars.pop()
            typedefs.append(new_char)
            output += 'TYPE_DEFINITION {} GES AMEND_PHASE_DESCRIPTION {} MAGNETIC {} {} !\n'\
                .format(new_char, name.upper(), model_hints['ihj_magnetic_afm_factor'],
                        model_hints['ihj_magnetic_structure_factor'])
            del model_hints['ihj_magnetic_afm_factor']
            del model_hints['ihj_magnetic_structure_factor']
        if len(model_hints) > 0:
            # Some model hints were not properly consumed
            raise ValueError('Not all model hints are supported: {}'.format(model_hints))
        output += "PHASE {0} {1}  {2} {3} !\n".format(name.upper(), ''.join(typedefs),
                                                      len(phase_obj.sublattices),
                                                      ' '.join([str(i) for i in phase_obj.sublattices]))
        constituents = ':'.join([','.join(sorted(subl)) for subl in phase_obj.constituents])
        output += "CONSTITUENT {0} :{1}: !\n".format(name.upper(), constituents)
        output += "\n"

    # PARAMETERs by subsystem
    param_sorted = defaultdict(lambda: list())
    paramtuple = namedtuple('ParamTuple', ['phase_name', 'parameter_type', 'complexity', 'constituent_array',
                                           'parameter_order', 'parameter', 'reference'])
    for param in dbf._parameters.all():
        components = set()
        for subl in param['constituent_array']:
            components |= set(subl)
        components = tuple(sorted([c.upper() for c in components]))
        # We use the complexity parameter to help with sorting the parameters logically
        param_sorted[components].append(paramtuple(param['phase_name'], param['parameter_type'],
                                                   sum([len(i) for i in param['constituent_array']]),
                                                   param['constituent_array'], param['parameter_order'],
                                                   param['parameter'], param['reference']))
    for num_elements in range(1, len(dbf.elements)+1):
        subsystems = list(itertools.combinations(sorted([i.upper() for i in dbf.elements]), num_elements))
        for subsystem in subsystems:
            parameters = sorted(param_sorted[subsystem])
            if len(parameters) > 0:
                output += "\n\n"
                output += "$" * 80 + "\n"
                output += "$ {}".format('-'.join(sorted(subsystem)).center(80, " ")[2:-1]) + "$\n"
                output += "$" * 80 + "\n"
                output += "\n"
            for parameter in parameters:
                constituents = ':'.join([','.join(sorted([i.upper() for i in subl]))
                                         for subl in parameter.constituent_array])
                # TODO: Handle references
                expr = TCPrinter().doprint(parameter.parameter).upper()
                if ';' not in expr:
                    expr += '; N'
                output += "PARAMETER {}({},{};{}) {} !\n".format(parameter.parameter_type.upper(),
                                                                 parameter.phase_name.upper(),
                                                                 constituents,
                                                                 parameter.parameter_order,
                                                                 expr)
    # Reflow text to respect 80 character limit per line
    fd.write(reflow_text(output, linewidth=80))


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
    lines = fd.read()
    lines = lines.replace('\t', ' ')
    lines = lines.strip()
    # Split the string by newlines
    splitlines = lines.split('\n')
    # Remove extra whitespace inside line
    splitlines = [' '.join(k.split()) for k in splitlines]
    # Remove comments
    splitlines = [k.strip().split('$', 1)[0] for k in splitlines]
    # Combine everything back together
    lines = ' '.join(splitlines)
    # Now split by the command delimeter
    commands = lines.split('!')
    # Filter out comments one more time
    # It's possible they were at the end of a command
    commands = [k.strip() for k in commands if not k.startswith("$")]

    # Temporary storage while we process type definitions
    dbf.tdbtypedefs = {}

    for command in commands:
        if len(command) == 0:
            continue
        tokens = None
        try:
            tokens = _tdb_grammar().parseString(command)
            _TDB_PROCESSOR[tokens[0]](dbf, *tokens[1:])
        except ParseException:
            print("Failed while parsing: " + command)
            print("Tokens: " + str(tokens))
            raise
    del dbf.tdbtypedefs


Database.register_format("tdb", read=read_tdb, write=write_tdb)
