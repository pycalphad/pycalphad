from sympy.core import Tuple
from sympy.core.basic import as_Basic
from sympy.logic.boolalg import Boolean, true, false
from sympy.utilities.misc import filldedent, func_name

# Removes ITE rewriting, which is not compatible with SymEngine
def exprcondpair_new(cls, expr, cond):
    expr = as_Basic(expr)
    if cond == True:
        return Tuple.__new__(cls, expr, true)
    elif cond == False:
        return Tuple.__new__(cls, expr, false)

    if not isinstance(cond, Boolean):
        raise TypeError(filldedent('''
            Second argument must be a Boolean,
            not `%s`''' % func_name(cond)))
    return Tuple.__new__(cls, expr, cond)
