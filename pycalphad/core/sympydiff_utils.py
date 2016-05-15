"""
This module constructs gradient functions for Models.
"""
from .custom_ufuncify import ufuncify
import numpy as np
import itertools

# Doesn't seem to be a run-time way to detect this, so we use the value as of numpy 1.11
_NPY_MAXARGS = 32

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    Source: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]

def make_gradient_from_graph(sympy_graph, variables):
    cflags = ['-ffast-math']
    wrt = tuple(variables)
    if len(wrt) > _NPY_MAXARGS:
        # TODO: Create a fallback mechanism
        raise ValueError('Cannot handle more than 32 degrees of freedom at once')
    grad_diffs = tuple(sympy_graph.diff(i) for i in wrt)
    hess_diffs = []
    for i in range(len(wrt)):
        for j in range(i, len(wrt)):
            hess_diffs.append(grad_diffs[i].diff(wrt[j]))
    # Chunking is necessary to work around NPY_MAXARGS limit in ufuncs, see numpy/numpy#4398
    grad = [ufuncify(wrt, gd, cflags=cflags) for gd in chunks(grad_diffs, _NPY_MAXARGS-len(wrt))]
    hess = [ufuncify(wrt, hd, cflags=cflags) for hd in chunks(hess_diffs, _NPY_MAXARGS-len(wrt))]

    # Factored out of argwrapper functions via profiling
    triu_indices = np.triu_indices(len(wrt))
    lenargsrange = np.arange(len(wrt), dtype=np.int)

    def grad_argwrapper(*args):
        result = np.array(list(itertools.chain(*(f(*args) for f in grad))))
        axes = tuple(range(len(result.shape)))
        # Send 'gradient' axis back
        result = result.transpose(axes[1:] + axes[:1])
        return result
    def hess_argwrapper(*args):
        resarray = list(itertools.chain(*(f(*args) for f in hess)))
        result = np.zeros((len(args), len(args)) + resarray[0].shape)
        result[triu_indices] = resarray
        axes = tuple(range(len(result.shape)))
        # Upper triangular is filled; also need to fill lower triangular
        result = result + result.swapaxes(0, 1)
        # Return diagonal to its original value since we doubled it above
        result[lenargsrange, lenargsrange] /= 2
        # Send 'Hessian' axes back
        result = result.transpose(axes[2:] + axes[:2])
        return result
    return grad_argwrapper, hess_argwrapper