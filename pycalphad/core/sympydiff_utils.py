"""
This module constructs gradient functions for Models.
"""
from sympy import lambdify
from pycalphad.core.utils import NumPyPrinter
import numba
import numpy as np

@numba.jit('float64(boolean, float64, float64)')
def where(condition, x, y):
    if condition:
        return x
    else:
        return y

def make_gradient_from_graph(mod):
    wrt = tuple(mod.variables)
    grads = np.empty((len(wrt)), dtype=object)
    #hess_indices = []
    namespace = {}
    for i, mgrad in zip(range(len(wrt)), mod.gradient):
        grads[i] = mgrad
        #for j in range(i, len(wrt)):
        #    namespace['hess_{0}{1}'.format(i, j)] = numba.vectorize(lambdify(tuple(wrt), grads[i].diff(wrt[j]), dummify=True,
        #                                                      modules=[{'where': nbwhere}, 'numpy'], printer=NumPyPrinter))
        #    hess_indices.append((i, j))
        namespace['grad_{0}'.format(i)] = numba.vectorize(lambdify(tuple(wrt), grads[i], dummify=True,
                                                    modules=[{'where': where}, 'numpy'], printer=NumPyPrinter))
    # Build the gradient and Hessian using compile() and exec
    # We do this because Numba needs "static" information about the arguments and functions
    call_args = ','.join(['_x{0}'.format(i) for i in range(len(wrt))])
    call_passed_args = ','.join(['_x{0}[0]'.format(i) for i in range(len(wrt))])

    grad_code = 'def grad_func({0}, lengthfix, result):'.format(call_args)
    grad_list = ['    result[{0}] = grad_{0}({1})'.format(i, call_passed_args) for i in range(len(wrt))]
    grad_code = grad_code + '\n' + '\n'.join(grad_list)

    grad_code = compile(grad_code, '<string>', 'exec')
    try:
        exec(grad_code, namespace)
    except SyntaxError:
        exec (grad_code in namespace)

    # Now construct the Hessian
    #hess_code = 'def hess_func({0}, lengthfix, result):'.format(call_args)
    #hess_list = ['    result[{0},{1}] = result[{1}, {0}] = hess_{0}{1}({2})'.format(i, j, call_passed_args) for i, j in hess_indices]
    #hess_code = hess_code + '\n' + '\n'.join(hess_list)
    #print(hess_code)

    #hess_code = compile(hess_code, '<string>', 'exec')
    #try:
    #    exec hess_code in namespace
    #except SyntaxError:
    #    exec(hess_code, namespace)

    grad_func = numba.guvectorize([','.join(['float64[:]'] * (len(wrt)+2))],
                                   ','.join(['()'] * len(wrt)) + ',(n)->(n)', nopython=True)(namespace['grad_func'])
    #hess_func = None
    #hess_func = numba.guvectorize([','.join(['float64[:]'] * (len(wrt)+1)) + ',float64[:,:]'],
    #                               ','.join(['()'] * len(wrt)) + ',(n)->(n,n)', nopython=True)(namespace['hess_func'])
    return grad_func#, hess_func