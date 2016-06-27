"""
This module constructs gradient functions for Models.
"""
from .custom_ufuncify import ufuncify
from .tempfilemanager import TempfileManager
from .autograd_utils import build_functions as interpreted_build_functions
from sympy import zoo, oo
import numpy as np
import itertools
import logging
import os

# Doesn't seem to be a run-time way to detect this, so we use the value as of numpy 1.11
_NPY_MAXARGS = 32

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    Source: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]

@TempfileManager(os.getcwd())
def build_functions(sympy_graph, variables, tmpman=None, include_obj=True, include_grad=True, include_hess=True):
    wrt = tuple(variables)
    if tmpman is None:
        raise ValueError('Missing temporary file context manager')
    if len(wrt) > _NPY_MAXARGS:
        logging.warning('Cannot handle more than NPY_MAXARGS degrees of freedom at once in compiled mode. '
                        'Falling back to interpreted.')
        return interpreted_build_functions(sympy_graph, variables, tmpman=tmpman, include_obj=include_obj,
                                           include_grad=include_grad, include_hess=include_hess)
    cflags = ['-ffast-math']
    flags = []
    restup = []
    grad = None
    hess = None
    if include_obj:
        restup.append(ufuncify(wrt, sympy_graph, tmpman=tmpman, flags=flags, cflags=cflags))
    if include_grad or include_hess:
        # Replacing zoo's is necessary because sympy's CCodePrinter doesn't handle them
        grad_diffs = tuple(sympy_graph.diff(i).xreplace({zoo: oo}) for i in wrt)
        hess_diffs = []
        # Chunking is necessary to work around NPY_MAXARGS limit in ufuncs, see numpy/numpy#4398
        if include_hess:
            for i in range(len(wrt)):
                for j in range(i, len(wrt)):
                    hess_diffs.append(grad_diffs[i].diff(wrt[j]).xreplace({zoo: oo}))
            hess = [ufuncify(wrt, hd, tmpman=tmpman, flags=flags, cflags=cflags)
                    for hd in chunks(hess_diffs, _NPY_MAXARGS - len(wrt))]
        if include_grad:
            grad = [ufuncify(wrt, gd, tmpman=tmpman, flags=flags, cflags=cflags)
                    for gd in chunks(grad_diffs, _NPY_MAXARGS-len(wrt))]

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
        if include_grad:
            restup.append(grad_argwrapper)
        if include_hess:
            restup.append(hess_argwrapper)
    if len(restup) == 1:
        return restup[0]
    else:
        return tuple(restup)