"""
This module defines functions for compiling symbolic SymEngine
expressions into fast callable functions.

The SymEngine ``lambdify`` function is used to compile the functions using a
particular backend, with or without common subexpression elimination (CSE).

By default, the LLVM backend is used with the ``opt_level=0`` option and common
subexpression elimination is on, as defined by the module constants::

    LAMBDIFY_DEFAULT_BACKEND = 'llvm'
    LAMBDIFY_DEFAULT_CSE = True
    LAMBDIFY_DEFAULT_LLVM_OPT_LEVEL = 0

Note that as of February 2020, SymEngine only supports using ``'lambda'`` or
``'llvm'`` backends. The LLVM backend uses the LLVM compiler to compile the
expressions, and is slower to build the callable functions than the Lambda
backend, though in principle the LLVM runtime performance can be better as
LLVM can optimize the functions.

Callables produced with the Lambda backend cannot be pickled, since SymEngine
does not define how its objects should be serialized. This is important to
packages that may want to parallelize pycalphad calls and pickle the callable
functions or PhaseRecords. The following issues track this behavior:

* SymEngine: https://github.com/symengine/symengine/issues/1394
* SymEngine.py: https://github.com/symengine/symengine.py/issues/294

"""
from pycalphad.core.cache import cacheit
from pycalphad.core.utils import wrap_symbol
import hashlib
from operator import itemgetter
from symengine import Add, Basic, ComplexDouble, Mul, Piecewise, RealDouble, sympify, lambdify, I, zoo, oo, have_llvm
from collections import namedtuple

_ComplexInfinity = type(zoo)
# SymEngine's own hash. pycalphad's Symbol subclasses override __hash__ with Python's string hash,
# which varies between processes (PYTHONHASHSEED); SymEngine's does not.
_symengine_hash = Basic.__hash__

BuildFunctionsResult = namedtuple('BuildFunctionsResult', ['func', 'grad', 'hess'])
ConstraintFunctions = namedtuple('ConstraintFunctions', ['cons_func', 'cons_jac', 'cons_hess'])

if have_llvm:
    LAMBDIFY_DEFAULT_BACKEND = 'llvm'
else:
    LAMBDIFY_DEFAULT_BACKEND = 'lambda'
LAMBDIFY_DEFAULT_CSE = True
LAMBDIFY_DEFAULT_LLVM_OPT_LEVEL = 0


def _get_lambidfy_options(user_options):
    user_options = user_options if user_options is not None else {}
    user_options.setdefault('cse', LAMBDIFY_DEFAULT_CSE)
    user_options.setdefault('backend', LAMBDIFY_DEFAULT_BACKEND)
    if user_options['backend'] == 'llvm':
        user_options.setdefault('opt_level', LAMBDIFY_DEFAULT_LLVM_OPT_LEVEL)
    return user_options


def _canonical_real_graph(expr):
    """Prepare ``expr`` for compilation: replace complex infinity with real infinity, drop the imaginary
    parts of complex numbers, and rebuild every sum and product with its terms in a canonical order.

    SymEngine cannot lambdify complex numbers, and integer exponents expressed as doubles can evaluate
    to negligible (1e-16*j) complex parts, which raise runtime errors for ``lambdify(..., real=True)``.
    Dropping them would also drop legitimate complex numbers (e.g. negative bases for non-integer
    exponents), but we have been doing this all along internally inside ``lambdify``.

    SymEngine stores the terms of ``Add`` and ``Mul`` in an order that depends on how an expression was
    built. Equality, hashing and printing ignore that order, but compiled functions evaluate the terms in
    it, and floating point addition and multiplication are not associative. Equal expressions built
    differently (e.g. from databases that write the same parameters differently) could then compile to
    functions that round differently, and because compiled functions are cached by equality, results
    could depend on what was compiled earlier in the process.

    Terms are ordered by their SymEngine hash, which depends only on their content (also between
    processes); ties, which are rare, are broken by a digest of the content. All of this is done in a single pass over the expression.
    """
    rebuilt = {}
    digests = {}

    def digest(e):
        key = digests.get(e)
        if key is None:
            if not e.args:
                text = float(e).hex() if isinstance(e, RealDouble) else str(e)
                key = hashlib.sha1(f'{type(e).__name__}:{text}'.encode()).hexdigest()
            else:
                children = [digest(a) for a in e.args]
                if isinstance(e, (Add, Mul)):
                    children.sort()
                key = hashlib.sha1(f'{type(e).__name__}({",".join(children)})'.encode()).hexdigest()
            digests[e] = key
        return key

    def rebuild(e):
        if isinstance(e, _ComplexInfinity):  # zoo has an argument, so check it before treating e as a leaf
            return oo
        args = e.args
        if not args:
            if isinstance(e, ComplexDouble):
                return e.xreplace({I: 0.0})
            return e
        out = rebuilt.get(e)
        if out is None:
            args = [rebuild(a) for a in args]
            if isinstance(e, (Add, Mul)):
                keys = [_symengine_hash(a) for a in args]
                if len(set(keys)) < len(keys):
                    keys = [(k, digest(a)) for k, a in zip(keys, args)]
                out = type(e)(*[a for _, a in sorted(zip(keys, args), key=itemgetter(0))])
            elif isinstance(e, Piecewise):
                out = Piecewise(*zip(args[0::2], args[1::2]))
            else:
                out = type(e)(*args)
            rebuilt[e] = out
        return out

    return rebuild(expr)


@cacheit
def build_functions(symengine_graph, variables, parameters=None, wrt=None,
                    include_obj=True, include_grad=False, include_hess=False,
                    func_options=None, grad_options=None, hess_options=None):
    r"""Build function, gradient, and Hessian callables of the symengine_graph.

    Parameters
    ----------
    symengine_graph : symengine.Basic
        symengine expression to compile,
        :math:`f(x) : \mathbb{R}^{n} \\rightarrow \mathbb{R}`,
        which will corresponds to ``symengine_graph(variables+parameters)``
    variables : List[symengine.Symbol]
        Free variables in the symengine_graph. By convention these are usually all
        instances of StateVariables.
    parameters : Optional[List[symengine.Symbol]]
        Free variables in the symengine_graph. These are typically external
        parameters that are controlled by the user.
    wrt : Optional[List[symengine.Symbol]]
        Variables to differentiate *with respect to* for the gradient and
        Hessian callables. If None, will fall back to ``variables``.
    include_obj : Optional[bool]
        Whether to build the symengine_graph callable,
        :math:`f(x) : \mathbb{R}^{n} \\rightarrow \mathbb{R}`
    include_grad : Optional[bool]
        Whether to build the gradient callable,
        :math:`\\pmb{g}(x) = \\nabla f(x) : \mathbb{R}^{n} \\rightarrow \mathbb{R}^{n}`
    include_hess : Optional[bool]
        Whether to build the Hessian callable,
        :math:`\mathbb{H}(x) = \\nabla^2 f(x) : \mathbb{R}^{n} \\rightarrow \mathbb{R}^{n \\times n}`
    func_options : Optional[Dict[str, str]]
        Options to pass to ``lambdify`` when compiling the function.
    grad_options : Optional[Dict[str, str]]
        Options to pass to ``lambdify`` when compiling the gradient.
    hess_options : Optional[Dict[str, str]]
        Options to pass to ``lambdify`` when compiling Hessian.

    Returns
    -------
    BuildFunctionsResult

    Notes
    -----
    Default options for compiling the function, gradient and Hessian are defined by ``_get_lambdify_options``.

    """
    if wrt is None:
        wrt = sympify(tuple(variables))
    if parameters is None:
        parameters = []
    else:
        parameters = [wrap_symbol(p) for p in parameters]
    variables = tuple(variables)
    parameters = tuple(parameters)
    func, grad, hess = None, None, None
    # Replace complex infinity (zoo) with real infinity because SymEngine
    # cannot lambdify complex numbers. We also replace in the derivatives in
    # case some differentiation would produce a complex infinity. The
    # replacement is assumed to be cheap enough that it's safer to replace the
    # complex values and pay the minor time penalty.
    inp = sympify(variables + parameters)
    graph = _canonical_real_graph(sympify(symengine_graph))
    func = lambdify(inp, [graph], **_get_lambidfy_options(func_options))
    if include_grad or include_hess:
        grad_graphs = list(graph.diff(w).xreplace({zoo: oo}) for w in wrt)
        if include_grad:
            grad = lambdify(inp, grad_graphs, **_get_lambidfy_options(grad_options))
        if include_hess:
            hess_graphs = list(list(g.diff(w).xreplace({zoo: oo}) for w in wrt) for g in grad_graphs)
            hess = lambdify(inp, hess_graphs, **_get_lambidfy_options(hess_options))
    return BuildFunctionsResult(func=func, grad=grad, hess=hess)


@cacheit
def build_constraint_functions(variables, constraints, parameters=None, func_options=None, jac_options=None, hess_options=None):
    """Build callables functions for the constraints, constraint Jacobian, and constraint Hessian.

    Parameters
    ----------
    variables : List[symengine.Symbol]
        Free variables in the symengine_graph. By convention these are usually all
        instances of StateVariables.
    constraints : List[symengine.Basic]
        List of SymEngine expression to compile
    parameters : Optional[List[symengine.Symbol]]
        Free variables in the symengine_graph. These are typically external
        parameters that are controlled by the user.
    func_options : None, optional
        Options to pass to ``lambdify`` when compiling the function.
    jac_options : Optional[Dict[str, str]]
        Options to pass to ``lambdify`` when compiling the Jacobian.
    hess_options : Optional[Dict[str, str]]
        Options to pass to ``lambdify`` when compiling Hessian.

    Returns
    -------
    ConstraintFunctions

    Notes
    -----
    Default options for compiling the function, gradient and Hessian are defined by ``_get_lambdify_options``.

    """
    if parameters is None:
        parameters = []
    else:
        parameters = [wrap_symbol(p) for p in parameters]
    variables = tuple(variables)
    wrt = variables
    parameters = tuple(parameters)
    constraint_func, jacobian_func, hessian_func = None, None, None
    # Replace complex infinity (zoo) with real infinity because SymEngine
    # cannot lambdify complex numbers. We also replace in the derivatives in
    # case some differentiation would produce a complex infinity. The
    # replacement is assumed to be cheap enough that it's safer to replace the
    # complex values and pay the minor time penalty.
    inp = sympify(variables + parameters)
    graph = sympify([_canonical_real_graph(sympify(f)) for f in constraints])
    constraint_func = lambdify(inp, [graph], **_get_lambidfy_options(func_options))

    grad_graphs = list(list(c.diff(w).xreplace({zoo: oo}) for w in wrt) for c in graph)
    jacobian_func = lambdify(inp, grad_graphs, **_get_lambidfy_options(jac_options))

    hess_graphs = list(list(list(g.diff(w).xreplace({zoo: oo}) for w in wrt) for g in c) for c in grad_graphs)
    hessian_func = lambdify(inp, hess_graphs, **_get_lambidfy_options(hess_options))
    return ConstraintFunctions(cons_func=constraint_func, cons_jac=jacobian_func, cons_hess=hessian_func)
