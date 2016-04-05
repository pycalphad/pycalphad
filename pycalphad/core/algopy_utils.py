"""
This module manages interactions with the algopy library.
"""

from pycalphad.core.utils import NumPyPrinter
from sympy import lambdify
from sympy.printing.lambdarepr import LambdaPrinter
import algopy
import numpy as np

def extract_hessian(N, y):
    H = np.zeros((y.data.shape[1], N,N), dtype=y.data.dtype)
    for n in range(N):
        for m in range(n):
            a =  sum(range(n+1))
            b =  sum(range(m+1))
            k =  sum(range(n+2)) - m - 1
            if n!=m:
                tmp = (y.data[2, :, k] - y.data[2, :, a] - y.data[2, :, b])
                H[:, m,n]= H[:, n,m]= tmp
        a =  sum(range(n+1))
        H[:, n,n] = 2*y.data[2, :, a]
    return H

class AlgoPyPrinter(LambdaPrinter):
    def _print_Piecewise(self, expr):
        result = []
        i = 0
        for arg in expr.args:
            e = arg.expr
            c = arg.cond
            result.append('((')
            result.append(self._print(e))
            result.append(') if (')
            result.append(self._print(c))
            result.append(') else (')
            i += 1
        result = result[:-1]
        # print zero instead of None so algopy is happy
        result.append(') else 0)')
        result.append(')' * (2 * i - 2))
        return ''.join(result)

def build_functions(sympy_graph, variables):
    trace_func = lambdify(tuple(variables), sympy_graph, dummify=True,
                          modules=[algopy], printer=AlgoPyPrinter)
    logical_np = [{'And': np.logical_and, 'Or': np.logical_or}, np]
    obj = lambdify(tuple(variables), sympy_graph, dummify=True,
                   modules=logical_np, printer=NumPyPrinter)

    def grad_func(*args):
        inp_arr = np.rollaxis(np.array(np.broadcast_arrays(*args), dtype=np.float), 0, 0)
        orig_shape = tuple(inp_arr.shape)
        inp_arr.shape = (-1, len(args))
        N = len(args)
        x = algopy.UTPM(np.zeros((2, inp_arr.shape[0], N,N)))
        x.data[0, :, :, :] = inp_arr[..., None, :]
        x.data[1, :, :] = np.eye(N)

        y = trace_func(*[x[..., i] for i in range(N)])
        return y.data[1, :, :].T.reshape(orig_shape[1:] + (N,))

    def hess_func(*args):
        inp_arr = np.rollaxis(np.array(np.broadcast_arrays(*args), dtype=np.float), 0, 0)
        orig_shape = tuple(inp_arr.shape)
        inp_arr.shape = (-1, len(args))
        # generate directions
        N = len(args)
        M = (N * (N + 1)) / 2
        S = np.zeros((N, M))

        s = 0
        i = 0
        for n in range(1, N + 1):
            S[-n:, s:s + n] = np.eye(n)
            S[-n, s:s + n] = np.ones(n)
            s += n
            i += 1
        S = S[::-1].T
        x = algopy.UTPM(np.zeros((3, inp_arr.shape[0]) + S.shape))
        x.data[0, :, :, :] = inp_arr[..., None, :]
        x.data[1, :, :] = S

        y = trace_func(*[x[..., i] for i in range(N)])
        return extract_hessian(N, y).reshape(orig_shape[1:] + (N,N))

    return obj, grad_func, hess_func