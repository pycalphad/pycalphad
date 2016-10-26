"""
This module constructs gradient functions for Models.
"""
from .custom_autowrap import autowrap
from .cache import cacheit
from sympy import zoo, oo, ImmutableMatrix, IndexedBase, Idx, Dummy, Lambda, Eq
import numpy as np
import os
import sys
import copy
import time
import tempfile

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    Source: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]

def import_module(path, modname):
    import importlib.util
    import glob
    npath = glob.glob(os.path.join(path, modname+'.*.so'))
    if len(npath) == 1:
        npath = npath[0]
    else:
        raise ImportError('Failed to import', os.path.join(path, modname+'.*.so'))
    spec = importlib.util.spec_from_file_location(modname, npath)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo


class PickleableFunction(object):
    """
    A compiled function that is recompiled when unpickled.
    This works around several issues with sending JIT'd functions over the wire.
    This approach means only the underlying SymPy object must be pickleable.
    This also means multiprocessing using fork() will NOT force a recompile.
    """
    def __init__(self, sympy_vars, sympy_obj):
        self._sympyobj = sympy_obj
        self._sympyvars = tuple(sympy_vars)
        self._workdir = tempfile.mkdtemp(prefix='pycalphad-')
        self._module_name = None
        self._routine_name = None
        self._kernel = None
        self._compiling = False

    @property
    def kernel(self):
        if self._kernel is None:
            if self._module_name is not None:
                start = time.time()
                mod = None
                while mod is None:
                    try:
                        mod = import_module(self._workdir, self._module_name)
                    except ImportError:
                        if start + 30 > time.time():
                            raise
                self._kernel = getattr(mod, self._routine_name)
            else:
                self._kernel = self.compile()
        return self._kernel

    def compile(self):
        raise NotImplementedError

    def __hash__(self):
        return hash((self._sympyobj, self._sympyvars))

    def __call__(self, *args, **kwargs):
        return self.kernel(*args, **kwargs)

    def __getstate__(self):
        # Explicitly drop the compiled function when pickling
        # The architecture of the unpickling machine may be incompatible with it
        return {key: value for key, value in self.__dict__.items() if str(key) != '_kernel'}

    def __setstate__(self, state):
        self._kernel = None
        for key, value in state.items():
            setattr(self, key, value)


class AutowrapFunction(PickleableFunction):
    def compile(self):
        # XXX: Acquire a thread lock here!
        result = autowrap(self._sympyobj, args=self._sympyvars, backend='f2py', language='F95', tempdir=self._workdir)
        self._module_name = str(result.module_name)
        self._routine_name = str(result.routine_name)
        return result


@cacheit
def build_functions(sympy_graph, variables, wrt=None, include_obj=True, include_grad=True, include_hess=True,
                    excluded=None):
    """

    Parameters
    ----------
    sympy_graph
    variables : tuple of Symbols
        Input arguments.
    wrt : tuple of Symbols, optional
        Variables to differentiate with respect to. (Default: equal to variables)
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
    restup = []
    grad = None
    hess = None
    if include_obj:
        if excluded:
            excluded = list(range(excluded))
        else:
            excluded = []
        y = IndexedBase(Dummy())
        m = Dummy(integer=True)
        i = Idx(Dummy(integer=True), m)
        # workaround for sympy/sympy#11692
        # that is why we don't use implemented_function
        from sympy import Function
        class f(Function):
            _imp_ = Lambda(variables, sympy_graph)
        # For each of the args create an indexed version.
        indexed_args = []
        for indx, a in enumerate(variables):
            if indx in excluded:
                indexed_args.append(a)
            else:
                indexed_args.append(IndexedBase(Dummy(a.name)))
        # Order the arguments (out, args, dim)
        args = [y] + indexed_args + [m]
        args_with_indices = []
        for indx, a in enumerate(indexed_args):
            if indx in excluded:
                args_with_indices.append(a)
            else:
                args_with_indices.append(a[i])
        restup.append(AutowrapFunction(args, Eq(y[i], f(*args_with_indices))))
    if include_grad or include_hess:
        # Replacing zoo's is necessary because sympy's CCodePrinter doesn't handle them
        grad_diffs = list(sympy_graph.diff(i).xreplace({zoo: oo}) for i in wrt)
        hess_diffs = []
        # Chunking is necessary to work around NPY_MAXARGS limit in ufuncs, see numpy/numpy#4398
        if include_hess:
            for i in range(len(wrt)):
                for j in range(i, len(wrt)):
                    hess_diffs.append(grad_diffs[i].diff(wrt[j]).xreplace({zoo: oo}))
            hess = AutowrapFunction(variables, ImmutableMatrix(hess_diffs))
        if include_grad:
            grad = AutowrapFunction(variables, ImmutableMatrix(grad_diffs))

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
