"""
This module constructs gradient functions for Models.
"""
from sympy import lambdify
from pycalphad.core.utils import NumPyPrinter
import numba
import numpy as np
import itertools

@numba.jit('float64(boolean, float64, float64)')
def where(condition, x, y):
    if condition:
        return x
    else:
        return y

def get_broadcast_shape(*shapes):
    """
    Given a set of array shapes, return the shape of the output when arrays of those
    shapes are broadcast together.
    Source: http://stackoverflow.com/questions/27196672/numpy-broadcast-indices-from-shape
    """
    max_nim = max(len(s) for s in shapes)
    equal_len_shapes = np.array([(1, )*(max_nim-len(s))+s for s in shapes])
    max_dim_shapes = np.max(equal_len_shapes, axis = 0)
    assert np.all(np.bitwise_or(equal_len_shapes==1, equal_len_shapes == max_dim_shapes[None, :])), \
        'Shapes %s are not broadcastable together' % (shapes, )
    return tuple(max_dim_shapes)

def make_gradient_from_graph(sympy_graph, variables):
    wrt = tuple(variables)
    grads = np.empty((len(wrt)), dtype=object)
    namespace = {}
    for i, mgrad in enumerate(sympy_graph.diff(vv) for vv in variables):
        grads[i] = mgrad
        gen_func = lambdify(tuple(wrt), grads[i], dummify=True, modules=[{'where': where,
                                                                          'Abs': np.abs}, 'numpy'], printer=NumPyPrinter)
        namespace['grad_{0}'.format(i)] = numba.jit(['float64({})'.format(','.join(['float64'] * len(wrt)))], nopython=True)\
            (gen_func)
    # Build the gradient using compile() and exec
    # We do this because Numba needs "static" information about the arguments and functions
    call_args = ','.join(['_x{0}'.format(i) for i in range(len(wrt))])
    call_passed_args = ','.join(['_x{0}[0]'.format(i) for i in range(len(wrt))])

    grad_code = 'def grad_func({0}, lengthfix, result):'.format(call_args)
    grad_list = ['    result[{0}] = grad_{0}({1})'.format(i, call_passed_args) for i in range(len(wrt))]
    grad_code = grad_code + '\n' + '\n'.join(grad_list)

    grad_code = compile(grad_code, '<string>', 'exec')
    exec(grad_code, namespace)
    grad_func = numba.guvectorize([','.join(['float64[:]'] * (len(wrt)+2))],
                                   ','.join(['()'] * len(wrt)) + ',(n)->(n)', nopython=True)(namespace['grad_func'])
    hess_code = 'def hess_func({0}, lengthfix, result):'.format(call_args)
    hess_list = []
    for i in range(len(wrt)):
        for j in range(i,len(wrt)):
            # We have to do a little bit of trickery here to make sure we don't exceed the domain of our variables
            finite_diff_args = ['_x{0}[0]'.format(g) for g in range(len(wrt))]
            finite_diff_args[j] = '_x{0}[0]+1e-14'.format(j)
            finite_diff_args = ','.join(finite_diff_args)
            hess_list.append('    result[{0},{1}] = result[{1},{0}] = ((grad_{0}({2}) - grad_{0}({3}))/1e-14)'
                             .format(i, j, finite_diff_args, call_passed_args))
    hess_code = hess_code + '\n' + '\n'.join(hess_list)
    hess_code = compile(hess_code, '<string>', 'exec')
    namespace['where'] = where
    exec(hess_code, namespace)
    hess_func = numba.guvectorize([','.join(['float64[:]'] * (len(wrt) + 1)) + ',float64[:,:]'],
                                  ','.join(['()'] * len(wrt)) + ',(n)->(n,n)', nopython=True)(namespace['hess_func'])
    def grad_argwrapper(*args, out=None):
        result_shape = get_broadcast_shape(*[np.asarray(arg).shape for arg in args])
        result_shape = result_shape + (len(args),)
        result = out if out is not None else np.zeros(result_shape, dtype=np.float)
        return grad_func(*list(itertools.chain(args, [np.empty(len(args)), result])))
    def hess_argwrapper(*args, out=None):
        result_shape = get_broadcast_shape(*[np.asarray(arg).shape for arg in args])
        result_shape = result_shape + (len(args),len(args))
        result = out if out is not None else np.zeros(result_shape, dtype=np.float)
        hess_func(*list(itertools.chain(args, [np.empty(len(args)), result])))
        return result
    return grad_argwrapper, hess_argwrapper