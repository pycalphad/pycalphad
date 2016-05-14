"""The tdb module provides support for reading and writing databases in
Thermo-Calc TDB format.
"""

from pyparsing import CaselessKeyword, CharsNotIn, Group
from pyparsing import LineEnd, MatchFirst, OneOrMore, Optional, Regex, SkipTo
from pyparsing import ZeroOrMore, Suppress, White, Word, alphanums, alphas, nums
from pyparsing import delimitedList, ParseException
import re
from sympy import sympify, And, Or, Not, Intersection, Union, EmptySet, Interval, Piecewise
from sympy import Symbol, GreaterThan, StrictGreaterThan, LessThan, StrictLessThan, Complement, S
from sympy import Mul, Pow, Rational
from sympy.printing.str import StrPrinter
from sympy.core.mul import _keep_coeff
from sympy.printing.precedence import precedence
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
    # matching float w/ regex is ugly but is recommended by pyparsing
    float_number = Regex(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?') \
        .setParseAction(lambda t: [float(t[0])])
    # symbol name, e.g., phase name, function name
    symbol_name = Word(alphanums+'_:', min=1)
    # species name, e.g., CO2, AL, FE3+
    species_name = Word(alphanums+'+-*', min=1) + Optional(Suppress('%'))
    # constituent arrays are colon-delimited
    # each subarray can be comma- or space-delimited
    constituent_array = Group(delimitedList(Group(OneOrMore(Optional(Suppress(',')) + species_name)), ':'))
    param_types = MatchFirst([TCCommand(param_type) for param_type in TDB_PARAM_TYPES])
    # Let sympy do heavy arithmetic / algebra parsing for us
    # a convenience function will handle the piecewise details
    func_expr = Optional(float_number) + OneOrMore(SkipTo(';') \
        + Suppress(';') + ZeroOrMore(Suppress(',')) + Optional(float_number) + \
        Suppress(Word('YNyn', exact=1) | White()))
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
        Suppress('(') + symbol_name + \
        Optional(Suppress('&') + Word(alphas+'/-', min=1, max=2), default=None) + \
        Suppress(',') + constituent_array + \
        Optional(Suppress(';') + int_number, default=0) + \
        Suppress(')') + func_expr.setParseAction(_make_piecewise_ast)
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
                if model_hints['disordered_phase'] in targetdb.phases:
                    targetdb.phases[model_hints['disordered_phase']]\
                        .model_hints.update({'ordered_phase': model_hints['ordered_phase'],
                                             'disordered_phase': model_hints['disordered_phase']})
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
                           param_order, param, ref, diffusing_species)

def _unimplemented(*args, **kwargs): #pylint: disable=W0613
    """
    Null function.
    """
    pass

def _setitem_raise_duplicates(dictionary, key, value):
    if key in dictionary:
        raise ValueError("TDB contains duplicate FUNCTION {}".format(key))
    dictionary[key] = value

_TDB_PROCESSOR = {
    'ELEMENT': lambda db, el: db.elements.add(el),
    'TYPE_DEFINITION': _process_typedef,
    'FUNCTION': lambda db, name, sym: _setitem_raise_duplicates(db.symbols, name, sym),
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

def to_interval(relational):
    if isinstance(relational, And):
        return Intersection([to_interval(i) for i in relational.args])
    elif isinstance(relational, Or):
        return Union([to_interval(i) for i in relational.args])
    elif isinstance(relational, Not):
        return Complement([to_interval(i) for i in relational.args])
    if relational == S.true:
        return Interval(S.NegativeInfinity, S.Infinity, left_open=True, right_open=True)

    if len(relational.free_symbols) != 1:
        raise ValueError('Relational must only have one free symbol')
    if len(relational.args) != 2:
        raise ValueError('Relational must only have two arguments')
    free_symbol = list(relational.free_symbols)[0]
    lhs = relational.args[0]
    rhs = relational.args[1]
    if isinstance(relational, GreaterThan):
        if lhs == free_symbol:
            return Interval(rhs, S.Infinity, left_open=False)
        else:
            return Interval(S.NegativeInfinity, rhs, right_open=False)
    elif isinstance(relational, StrictGreaterThan):
        if lhs == free_symbol:
            return Interval(rhs, S.Infinity, left_open=True)
        else:
            return Interval(S.NegativeInfinity, rhs, right_open=True)
    elif isinstance(relational, LessThan):
        if lhs != free_symbol:
            return Interval(rhs, S.Infinity, left_open=False)
        else:
            return Interval(S.NegativeInfinity, rhs, right_open=False)
    elif isinstance(relational, StrictLessThan):
        if lhs != free_symbol:
            return Interval(rhs, S.Infinity, left_open=True)
        else:
            return Interval(S.NegativeInfinity, rhs, right_open=True)
    else:
        raise ValueError('Unsupported Relational: {}'.format(relational.__class__.__name__))

class TCPrinter(StrPrinter):
    """
    Prints Thermo-Calc style function expressions.
    """
    def _print_Piecewise(self, expr):
        # Filter out default zeros since they are implicit in a TDB
        filtered_args = [i for i in expr.args if not ((i.cond == S.true) and (i.expr == S.Zero))]
        exprs = [self._print(arg.expr) for arg in filtered_args]
        # Only a small subset of piecewise functions can be represented
        # Need to verify that each cond's highlim equals the next cond's lowlim
        # to_interval() is used instead of sympy.Relational.as_set() for performance reasons
        intervals = [to_interval(i.cond) for i in filtered_args]
        if (len(intervals) > 1) and Intersection(intervals) != EmptySet():
            raise ValueError('Overlapping intervals cannot be represented: {}'.format(intervals))
        if not isinstance(Union(intervals), Interval):
            raise ValueError('Piecewise intervals must be continuous')
        if not all([arg.cond.free_symbols == {v.T} for arg in filtered_args]):
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

    def _print_Mul(self, expr):
        "Copied from sympy StrPrinter and modified to remove division."

        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        if len(b) == 0:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            # Thermo-Calc's parser can't handle division operators
            return sign + '*'.join(a_str) + "*%s" % self.parenthesize(b[0]**(-1), prec)
        else:
            # TODO: Make this Thermo-Calc compatible by removing division operation
            return sign + '*'.join(a_str) + "/(%s)" % '*'.join(b_str)

    def _print_Pow(self, expr, rational=False):
        "Copied from sympy StrPrinter to remove TC-incompatible Pow simplifications."
        PREC = precedence(expr)

        e = self.parenthesize(expr.exp, PREC)
        if self.printmethod == '_sympyrepr' and expr.exp.is_Rational and expr.exp.q != 1:
            # the parenthesized exp should be '(Rational(a, b))' so strip parens,
            # but just check to be sure.
            if e.startswith('(Rational'):
                return '%s**%s' % (self.parenthesize(expr.base, PREC), e[1:-1])
        return '%s**%s' % (self.parenthesize(expr.base, PREC), e)

    def _print_Infinity(self, expr):
        # Use "default value" though TC's Database Checker complains about this
        return ","

    def _print_Symbol(self, expr):
        if isinstance(expr, v.StateVariable):
            return expr.name
        else:
            # Thermo-Calc likes symbol references to be marked with a '#' at the end
            return expr.name + "#"

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
    linebreak_chars = [" ", "$"]
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
                if "$" in line:
                    # previous line was a comment
                    line = "$ " + line[linebreak_idx:]
                else:
                    # Always put some leading spaces at the start of a new line
                    # Otherwise TC may misunderstand the expression
                    line = "  " + line[linebreak_idx:]
            output_lines.append(line)
    return "\n".join(output_lines)


def write_tdb(dbf, fd, groupby='subsystem'):
    """
    Write a TDB file from a pycalphad Database object.

    Parameters
    ----------
    dbf : Database
        A pycalphad Database.
    fd : file-like
        File descriptor.
    groupby : ['subsystem', 'phase'], optional
        Desired grouping of parameters in the file.
    """
    writetime = datetime.datetime.now()
    maxlen = 78
    output = ""
    # Comment header block
    # Import here to prevent circular imports
    from pycalphad import __version__
    output += ("$" * maxlen) + "\n"
    output += "$ Date: {}\n".format(writetime.strftime("%Y-%m-%d %H:%M"))
    output += "$ Components: {}\n".format(', '.join(sorted(dbf.elements)))
    output += "$ Phases: {}\n".format(', '.join(sorted(dbf.phases.keys())))
    output += "$ Generated by {} (pycalphad {})\n".format(getpass.getuser(), __version__)
    output += ("$" * maxlen) + "\n\n"
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
        output += "PHASE {0} {1}  {2} {3} !\n".format(name.upper(), ''.join(typedefs[name]),
                                                      len(phase_obj.sublattices),
                                                      ' '.join([str(i) for i in phase_obj.sublattices]))
        constituents = ':'.join([','.join(sorted(subl)) for subl in phase_obj.constituents])
        output += "CONSTITUENT {0} :{1}: !\n".format(name.upper(), constituents)
        output += "\n"

    # PARAMETERs by subsystem
    param_sorted = defaultdict(lambda: list())
    paramtuple = namedtuple('ParamTuple', ['phase_name', 'parameter_type', 'complexity', 'constituent_array',
                                           'parameter_order', 'diffusing_species', 'parameter', 'reference'])
    for param in dbf._parameters.all():
        if groupby == 'subsystem':
            components = set()
            for subl in param['constituent_array']:
                components |= set(subl)
            if param['diffusing_species'] is not None:
                components |= {param['diffusing_species']}
            # Wildcard operator is not a component
            components -= {'*'}
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
        constituents = ':'.join([','.join(sorted([i.upper() for i in subl]))
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
        if param_to_write.diffusing_species is not None:
            ds = "&" + param_to_write.diffusing_species
        else:
            ds = ""
        return "PARAMETER {}({}{},{};{}) {} !\n".format(param_to_write.parameter_type.upper(),
                                                        param_to_write.phase_name.upper(),
                                                        ds,
                                                        constituents,
                                                        param_to_write.parameter_order,
                                                        exprx)
    if groupby == 'subsystem':
        for num_elements in range(1, 5):
            subsystems = list(itertools.combinations(sorted([i.upper() for i in dbf.elements]), num_elements))
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
        if len(dbf.elements) > 4:
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
