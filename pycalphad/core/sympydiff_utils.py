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

class LazyPickleableFunction:
    """
    A lazily-compiled function that is recompiled when unpickled and called.
    This works around several issues with sending JIT'd functions over the wire.
    This approach means only the underlying SymPy object must be pickleable.
    This also means multiprocessing using fork() will NOT force a recompile.
    """
    def __init__(self, sympy_vars, sympy_obj, kernel=None):
        if kernel is not None:
            self._kernel = kernel
        self._sympyobj = sympy_obj
        self._sympyvars = sympy_vars

    @property
    def kernel(self):
        if not hasattr(self, '_kernel'):
            self._kernel = self.compile()
        return self._kernel

    def compile(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.kernel(*args, **kwargs)

    def __getstate__(self):
        # Explicitly drop the compiled function when pickling
        # The architecture of the unpickling machine may be incompatible with it
        return {key: value for key, value in self.__dict__.items() if str(key) != '_kernel'}

class UfuncifyFunction(LazyPickleableFunction):
    def __init__(self, sympy_vars, sympy_obj, tmpman=None, kernel=None):
        super(UfuncifyFunction, self).__init__(sympy_vars, sympy_obj, kernel=kernel)
        self.tmpman = tmpman
    def compile(self):
        return ufuncify(self._sympyvars, self._sympyobj, tmpman=self.tmpman, flags=[], cflags=['-ffast-math'])

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
    restup = []
    grad = None
    hess = None
    if include_obj:
        restup.append(UfuncifyFunction(wrt, sympy_graph, tmpman=tmpman))
    if include_grad or include_hess:
        # Replacing zoo's is necessary because sympy's CCodePrinter doesn't handle them
        grad_diffs = tuple(sympy_graph.diff(i).xreplace({zoo: oo}) for i in wrt)
        hess_diffs = []
        # Chunking is necessary to work around NPY_MAXARGS limit in ufuncs, see numpy/numpy#4398
        if include_hess:
            for i in range(len(wrt)):
                for j in range(i, len(wrt)):
                    hess_diffs.append(grad_diffs[i].diff(wrt[j]).xreplace({zoo: oo}))
            hess = [UfuncifyFunction(wrt, hd, tmpman=tmpman)
                    for hd in chunks(hess_diffs, _NPY_MAXARGS - len(wrt))]
        if include_grad:
            grad = [UfuncifyFunction(wrt, gd, tmpman=tmpman)
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