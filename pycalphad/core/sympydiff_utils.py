"""
This module constructs gradient functions for Models.
"""
from .custom_autowrap import autowrap, import_extension
from .cache import cacheit
from sympy import zoo, oo, ImmutableMatrix, IndexedBase, MatrixSymbol, Symbol, Idx, Dummy, Lambda, Eq, S
import numpy as np
import time
import tempfile
from threading import RLock
CompileLock = RLock()

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
    def __init__(self, sympy_vars, sympy_obj):
        self._sympyobj = sympy_obj
        self._sympyvars = tuple(sympy_vars)
        self._workdir = tempfile.mkdtemp(prefix='pycalphad-')
        self._module_name = None
        self._routine_name = None
        self._kernel = None
        self._cpointer = None

    @property
    def kernel(self):
        with CompileLock:
            if self._kernel is None:
                if self._module_name is not None:
                    start = time.time()
                    mod = None
                    while mod is None:
                        try:
                            mod = import_extension(self._workdir, self._module_name)
                            self._kernel = getattr(mod, self._routine_name + '_c')
                            self._cpointer = getattr(mod, 'get_pointer_c')()
                        except ImportError:
                            if start + 60 > time.time():
                                raise
                else:
                    self._kernel = self.compile()
        return self._kernel

    def compile(self):
        raise NotImplementedError

    def __hash__(self):
        return hash((self._sympyobj, self._sympyvars, self._workdir, self._routine_name, self._module_name))

    def __call__(self, inp, *args, **kwargs):
        # XXX: Hardcode until code rewrite is finished
        return self.kernel(inp, 0, *args, **kwargs)

    def __getstate__(self):
        # Explicitly drop the compiled function when pickling
        # The architecture of the unpickling machine may be incompatible with it
        return {key: value for key, value in self.__dict__.items() if str(key) not in ['_kernel', '_cpointer']}

    def __setstate__(self, state):
        self._kernel = None
        for key, value in state.items():
            setattr(self, key, value)


class AutowrapFunction(PickleableFunction):
    def compile(self):
        with CompileLock:
            result, self._cpointer, self._module_name, self._routine_name = autowrap(self._sympyobj, args=self._sympyvars, backend='Cython', language='C', tempdir=self._workdir)
        return result


@cacheit
def build_functions(sympy_graph, variables, wrt=None, include_obj=True, include_grad=True,
                    parameters=None):
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
    parameters

    Returns
    -------
    One or more functions.
    """
    if wrt is None:
        wrt = tuple(variables)
    if parameters is None:
        parameters = []
    new_parameters = []
    for param in parameters:
        if isinstance(param, Symbol):
            new_parameters.append(param)
        else:
            new_parameters.append(Symbol(param))
    parameters = tuple(new_parameters)
    variables = tuple(variables)
    restup = []
    grad = None
    hess = None
    m = Symbol('veclen', integer=True)
    i = Idx(Symbol('vecidx', integer=True), m)
    y = IndexedBase(Symbol('outp'))
    params = MatrixSymbol('params', 1, len(parameters))
    inp = MatrixSymbol('inp', m, len(variables))
    inp_nobroadcast = MatrixSymbol('inp', 1, len(variables))

    # workaround for sympy/sympy#11692
    # that is why we don't use implemented_function
    from sympy import Function
    class f(Function):
        _imp_ = Lambda(variables+parameters, sympy_graph.xreplace({zoo: oo, S.Pi: 3.14159265359}))
    args_with_indices = []
    args_nobroadcast = []
    for indx in range(len(variables)):
        args_with_indices.append(inp[i, indx])
        args_nobroadcast.append(inp_nobroadcast[0, indx])
    for indx in range(len(parameters)):
        args_with_indices.append(params[0, indx])
        args_nobroadcast.append(params[0, indx])

    args = [y, inp, params, m]
    if include_obj:
        restup.append(AutowrapFunction(args, Eq(y[i], f(*args_with_indices))))
    if include_grad:
        diffargs = (inp_nobroadcast, params)
        nobroadcast = dict(zip(variables+parameters, args_nobroadcast))
        sympy_graph_nobroadcast = sympy_graph.xreplace(nobroadcast)
        with CompileLock:
            grad_diffs = list(sympy_graph_nobroadcast.diff(nobroadcast[i]) for i in wrt)
        if include_grad:
            grad = AutowrapFunction(diffargs, ImmutableMatrix(grad_diffs))

        if include_grad:
            restup.append(grad)
    if len(restup) == 1:
        return restup[0]
    else:
        return tuple(restup)
