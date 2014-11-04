"""
Temporary patch for SymPy's theano_function
"""

# monkey patch for theano_function handling sympy
import sympy.printing.theanocode
import theano.tensor
import sympy
sympy.printing.theanocode.mapping[sympy.And] = theano.tensor.and_
def _special_print_Piecewise(self, expr, **kwargs):
    import numpy.nan
    from theano.ifelse import ifelse
    e, cond = expr.args[0].args
    if len(expr.args) == 1:
        return ifelse(self._print(cond, **kwargs),
                      self._print(e, **kwargs),
                      numpy.nan)
    return ifelse(self._print(cond, **kwargs),
                  self._print(e, **kwargs),
                  self._print(sympy.Piecewise(*expr.args[1:]), **kwargs))
sympy.printing.theanocode._print_Piecewise = _special_print_Piecewise

from sympy.printing.theanocode import theano_function
