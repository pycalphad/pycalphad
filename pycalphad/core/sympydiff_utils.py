"""
This module constructs gradient functions for Models.
"""
from .custom_ufuncify import ufuncify
from .tempfilemanager import TempfileManager
from .custom_autowrap import autowrap
from sympy import zoo, oo, ImmutableMatrix
import numpy as np
import os


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    Source: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]


class PickleableFunction(object):
    """
    A compiled function that is recompiled when unpickled.
    This works around several issues with sending JIT'd functions over the wire.
    This approach means only the underlying SymPy object must be pickleable.
    This also means multiprocessing using fork() will NOT force a recompile.
    """
    def __init__(self, sympy_vars, sympy_obj, kernel=None):
        self._sympyobj = sympy_obj
        self._sympyvars = sympy_vars
        if kernel is not None:
            self._kernel = kernel
        else:
            self._kernel = self.compile()
        self._compiling = False

    @property
    def kernel(self):
        return self._kernel

    def compile(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.kernel(*args, **kwargs)

    def __getstate__(self):
        # Explicitly drop the compiled function when pickling
        # The architecture of the unpickling machine may be incompatible with it
        return {key: value for key, value in self.__dict__.items() if str(key) != '_kernel'}

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        self._kernel = self.compile()


class UfuncifyFunction(PickleableFunction):
    def __init__(self, sympy_vars, sympy_obj, tmpman=None, kernel=None):
        self.tmpman = tmpman
        super(UfuncifyFunction, self).__init__(sympy_vars, sympy_obj, kernel=kernel)

    def compile(self):
        return ufuncify(self._sympyvars, self._sympyobj, tmpman=self.tmpman, flags=[], cflags=['-ffast-math'])


class AutowrapFunction(PickleableFunction):
    def __init__(self, sympy_vars, sympy_obj, tmpman=None, kernel=None):
        self.tmpman = tmpman
        super(AutowrapFunction, self).__init__(sympy_vars, sympy_obj, kernel=kernel)

    def compile(self):
        return autowrap(self._sympyobj, args=self._sympyvars, backend='f2py', language='F95')


@TempfileManager(os.getcwd())
def build_functions(sympy_graph, variables, wrt=None, tmpman=None, include_obj=True, include_grad=True, include_hess=True,
                    excluded=None):
    """

    Parameters
    ----------
    sympy_graph
    variables : tuple of Symbols
        Input arguments.
    wrt : tuple of Symbols, optional
        Variables to differentiate with respect to. (Default: equal to variables)
    tmpman
    include_obj
    include_grad
    include_hess
    excluded

    Returns
    -------
    One or more functions.
    """
    if wrt is None:
        wrt = tuple(variables)
    variables = tuple(variables)
    if tmpman is None:
        raise ValueError('Missing temporary file context manager')
    restup = []
    grad = None
    hess = None
    if include_obj:
        if excluded:
            excluded = list(range(excluded))
        restup.append(np.vectorize(AutowrapFunction(variables, sympy_graph, tmpman=tmpman), excluded=excluded))
    if include_grad or include_hess:
        # Replacing zoo's is necessary because sympy's CCodePrinter doesn't handle them
        grad_diffs = list(sympy_graph.diff(i).xreplace({zoo: oo}) for i in wrt)
        hess_diffs = []
        # Chunking is necessary to work around NPY_MAXARGS limit in ufuncs, see numpy/numpy#4398
        if include_hess:
            for i in range(len(wrt)):
                for j in range(i, len(wrt)):
                    hess_diffs.append(grad_diffs[i].diff(wrt[j]).xreplace({zoo: oo}))
            hess = AutowrapFunction(variables, ImmutableMatrix(hess_diffs), tmpman=tmpman)
        if include_grad:
            grad = AutowrapFunction(variables, ImmutableMatrix(grad_diffs), tmpman=tmpman)

        # Factored out of argwrapper functions via profiling
        triu_indices = np.triu_indices(len(wrt))
        lenargsrange = np.arange(len(wrt), dtype=np.int)

        def grad_argwrapper(*args):
            result = grad(*args)
            axes = tuple(range(len(result.shape)))
            # Send 'gradient' axis back
            result = result.transpose(axes[1:] + axes[:1])
            return result

        def hess_argwrapper(*args):
            resarray = hess(*args)
            result = np.zeros((len(wrt), len(wrt)) + resarray[0].shape)
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
