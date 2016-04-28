"""
This module manages interactions with the autograd library.
"""
import autograd.numpy as anp
from autograd import elementwise_grad, jacobian
from sympy import lambdify
from pycalphad.core.utils import NumPyPrinter
from itertools import chain


def elementwise_hess(fun, argnum=0):
    """
    From https://github.com/HIPS/autograd/issues/60
    """
    def sum_latter_dims(x):
        return anp.sum(x.reshape(x.shape[0], -1), 1)

    def sum_grad_output(*args, **kwargs):
        return sum_latter_dims(elementwise_grad(fun)(*args, **kwargs))
    return jacobian(sum_grad_output, argnum)


def build_functions(sympy_graph, variables):
    logical_np = [{'And': anp.logical_and, 'Or': anp.logical_or, 'Abs': anp.abs}, anp]
    obj = lambdify(tuple(variables), sympy_graph, dummify=True,
                   modules=logical_np, printer=NumPyPrinter)

    def argwrapper(args):
        return obj(*args)

    def grad_func(*args):
        inp = anp.array(anp.broadcast_arrays(*args))
        result = anp.atleast_2d(elementwise_grad(argwrapper)(inp))
        # Put 'gradient' axis at end
        axes = list(range(len(result.shape)))
        result = result.transpose(*chain(axes[1:], [axes[0]]))
        return result

    def hess_func(*args):
        result = anp.atleast_3d(elementwise_hess(argwrapper)(anp.array(anp.broadcast_arrays(*args))))
        # Put 'hessian' axes at end
        axes = list(range(len(result.shape)))
        result = result.transpose(*chain(axes[2:], axes[0:2]))
        return result

    return obj, grad_func, hess_func
