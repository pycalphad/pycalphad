# fourFn.py
#
# Demonstration of the pyparsing module, implementing a simple 4-function expression parser,
# with support for scientific notation, and symbols for e and pi.
# Extended to add exponentiation and simple built-in functions.
# Extended test cases, simplified pushFirst method.
# Removed unnecessary expr.suppress() call (thanks Nathaniel Peterson!), and added Group
# Changed fnumber to use a Regex, which is now the preferred method
# Reformatted to latest pypyparsing features, support multiple and variable args to functions
#
# Copyright 2003-2019 by Paul McGuire
#
from cPyparsing import (
    Literal,
    Word,
    Group,
    Forward,
    alphas,
    alphanums,
    Regex,
    ParseException,
    CaselessKeyword,
    Suppress,
    delimitedList,
)
import operator
import sympy
from pycalphad.io.grammar import float_number

exprStack = []


def push_first(toks):
    exprStack.append(toks[0])


def push_unary_minus(toks):
    for t in toks:
        if t == "-":
            exprStack.append("unary -")
        else:
            break


bnf = None


def BNF():
    """
    expop   :: '^'
    multop  :: '*' | '/'
    addop   :: '+' | '-'
    integer :: ['+' | '-'] '0'..'9'+
    atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
    factor  :: atom [ expop factor ]*
    term    :: factor [ multop factor ]*
    expr    :: term [ addop term ]*
    """
    global bnf
    if not bnf:
        # use CaselessKeyword for e and pi, to avoid accidentally matching
        # functions that start with 'e' or 'pi' (such as 'exp'); Keyword
        # and CaselessKeyword only match whole words
        T = CaselessKeyword("T")
        P = CaselessKeyword("P")
        # fnumber = Combine(Word("+-"+nums, nums) +
        #                    Optional("." + Optional(Word(nums))) +
        #                    Optional(e + Word("+-"+nums, nums)))
        # or use provided pyparsing_common.number, but convert back to str:
        # fnumber = ppc.number().addParseAction(lambda t: str(t[0]))
        fnumber = float_number
        ident = Word(alphas, alphanums + "_$")

        plus, minus, mult, div = map(Literal, "+-*/")
        lpar, rpar = map(Suppress, "()")
        addop = plus | minus
        multop = mult | div
        expop = Literal("**")

        expr = Forward()
        expr_list = delimitedList(Group(expr))
        # add parse action that replaces the function identifier with a (name, number of args) tuple
        def insert_fn_argcount_tuple(t):
            fn = t.pop(0)
            num_args = len(t[0])
            t.insert(0, (fn, num_args))

        fn_call = (ident + lpar - Group(expr_list) + rpar).setParseAction(
            insert_fn_argcount_tuple
        )
        atom = (
            addop[...]
            + (
                (fn_call | T | P | fnumber | ident).setParseAction(push_first)
                | Group(lpar + expr + rpar)
            )
        ).setParseAction(push_unary_minus)

        # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left
        # exponents, instead of left-to-right that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor <<= atom + (expop + factor).setParseAction(push_first)[...]
        term = factor + (multop + factor).setParseAction(push_first)[...]
        expr <<= term + (addop + term).setParseAction(push_first)[...]
        bnf = expr
    return bnf


# map operator symbols to corresponding arithmetic operations
epsilon = 1e-12

fn = {
    "exp": sympy.exp,
    "log": sympy.log,
    "ln": sympy.log
}

sym_T = sympy.Symbol('T')
sym_P = sympy.Symbol('P')

def evaluate_stack(s):
    op, num_args = s.pop(), 0
    if isinstance(op, tuple):
        op, num_args = op
    if op == "unary -":
        return -evaluate_stack(s)
    if op == '+':
        # note: operands are pushed onto the stack in reverse order
        op2 = evaluate_stack(s)
        op1 = evaluate_stack(s)
        return sympy.Add(op1, op2)
    elif op == '-':
        # note: operands are pushed onto the stack in reverse order
        op2 = evaluate_stack(s)
        op1 = evaluate_stack(s)
        return sympy.Add(op1, -op2)
    elif op == '*':
        # note: operands are pushed onto the stack in reverse order
        op2 = evaluate_stack(s)
        op1 = evaluate_stack(s)
        return sympy.Mul(op1, op2)
    elif op == '/':
        # note: operands are pushed onto the stack in reverse order
        op2 = evaluate_stack(s)
        op1 = evaluate_stack(s)
        return sympy.Mul(op1, 1/op2)
    elif op == '**':
        # note: operands are pushed onto the stack in reverse order
        op2 = evaluate_stack(s)
        op1 = evaluate_stack(s)
        return sympy.Pow(op1, op2)
    elif op == "T":
        return sym_T
    elif op == "P":
        return sym_P
    elif op in fn:
        # note: args are pushed onto the stack in reverse order
        args = reversed([evaluate_stack(s) for _ in range(num_args)])
        return fn[op](*args)
    else:
        try:
            val = float(op)
        except ValueError:
            return sympy.Symbol(op)
        if val.is_integer():
            return sympy.Integer(val)
        else:
            return sympy.Float(val)


if __name__ == "__main__":

    def test(s, expected):
        exprStack[:] = []
        try:
            results = BNF().parseString(s, parseAll=True)
            val = evaluate_stack(exprStack[:])
        except ParseException as pe:
            print(s, "failed parse:", str(pe))
        except Exception as e:
            print(s, "failed eval:", str(e), exprStack)
        else:
            if val == expected:
                print(s, "=", val, results, "=>", exprStack)
            else:
                print(s + "!!!", val, "!=", expected, results, "=>", exprStack)

    #test("-7976.15+137.093038*T", 9)
    # -7976.15+137.093038*T-24.3671976*T*ln(T) -1.884662E-3*T**2-0.877664E-6*T**3+74092*T**(-1)
    test("-7976.15+137.093038*T-24.3671976*T*ln(T) -1.884662E-3*T**2", -7976.15+137.093038*sym_T-24.3671976*sym_T*sympy.log(sym_T) -1.884662E-3*sym_T**2)
    test("-7976.15+137.093038*T-24.3671976*T*ln(T) -1.884662E-3*T**2-0.877664E-6*T**3+74092*T**(-1)", -7976.15+137.093038*sym_T-24.3671976*sym_T*sympy.log(sym_T) -1.884662E-3*sym_T**2-0.877664E-6*sym_T**3+74092*sym_T**(-1))
    #test("--9", 9)
    #test("9 + 3 + 6", 9 + 3 + 6)
    #test("9 + 3 / 11", 9 + 3.0 / 11)
    #test("(9 + 3)", (9 + 3))
    #test("(9+3) / 11", (9 + 3.0) / 11)
    #test("9 - 12 - 6", 9 - 12 - 6)
    #test("9 - (12 - 6)", 9 - (12 - 6))
    #test("2*3.14159", 2 * 3.14159)
    #test("3.1415926535*3.1415926535 / 10", 3.1415926535 * 3.1415926535 / 10)
    #test("3+T", 3+sym_T)
