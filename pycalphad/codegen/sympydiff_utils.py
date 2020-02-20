"""
This module defines functions compiling symbolic SymPy/SymEngine expressions
into fast callable functions.

The SymEngine ``lambdify`` function is used to compile the functions using a
particular backend, with or without common subexpressions eimination (CSE).

By default, the LLVM backend is used and common subexpression elimination is
on, as defined by the module constants::

    LAMBDIFY_DEFAULT_BACKEND = 'llvm'
    LAMBDIFY_DEFAULT_CSE = True

Note that as of February 2020, SymEngine only supports using ``'lambda'`` or
``'llvm'`` backends. The LLVM backend uses the LLVM compiler to compile the
expressions, and is slower to build the callable functions than the Lambda
backend, though in principle the LLVM runtime performance can is better as
LLVM can optimize the functions.

Additionally, callables from the Lambda backend cannot be pickled, since
SymEngine does not define how its object should be serialized. The following
issues track this behavior:

* SymEngine: https://github.com/symengine/symengine/issues/1394
* SymEngine.py: https://github.com/symengine/symengine.py/issues/294

"""
from pycalphad.core.cache import cacheit
from pycalphad.core.utils import wrap_symbol_symengine
from symengine import sympify, lambdify
from collections import namedtuple

BuildFunctionsResult = namedtuple('BuildFunctionsResult', ['func', 'grad', 'hess'])
ConstraintFunctions = namedtuple('ConstraintFunctions', ['cons_func', 'cons_jac', 'cons_hess'])

LAMBDIFY_DEFAULT_BACKEND = 'llvm'
LAMBDIFY_DEFAULT_CSE = True


def _get_lambidfy_options(user_options):
    default_options = {
        'backend': LAMBDIFY_DEFAULT_BACKEND,
        'cse': LAMBDIFY_DEFAULT_CSE
    }
    if user_options is not None:
        default_options.update(user_options)
    return default_options


@cacheit
def build_functions(sympy_graph, variables, parameters=None, wrt=None,
                    include_obj=True, include_grad=False, include_hess=False,
                    func_options=None, grad_options=None, hess_options=None):
    """Build function, gradient, and Hessian callables of the sympy_graph.

    Parameters
    ----------
    sympy_graph : sympy.core.expr.Expr
        SymPy expression to compile,
        :math:`f(x) : \mathbb{R}^{n} \\rightarrow \mathbb{R}`,
        which will be called by ``sympy_graph(variables+parameters)``
    variables : List[sympy.core.symbol.Symbol]
        Free variables in the sympy_graph. By convention these are usually all
        StateVariables instances.
    parameters : Optional[List[sympy.core.symbol.Symbol]]
        Free variables in the sympy_graph. These are typically external
        parameters that are controlled by the user.
    wrt : Optional[List[sympy.core.symbol.Symbol]]
        Variables to differentiate *with respect to* for gradient and Hessian
        callables. If None, the default is to differentiate w.r.t. all variables.
    include_obj : Optional[bool]
        Whether to build the sympy_graph callable,
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
        parameters = [wrap_symbol_symengine(p) for p in parameters]
    variables = tuple(variables)
    parameters = tuple(parameters)
    func, grad, hess = None, None, None
    inp = sympify(variables + parameters)
    graph = sympify(sympy_graph)
    # TODO: did not replace zoo with oo
    func = lambdify(inp, [graph], **_get_lambidfy_options(func_options))
    if include_grad or include_hess:
        grad_graphs = list(graph.diff(w) for w in wrt)
        if include_grad:
            grad = lambdify(inp, grad_graphs, **_get_lambidfy_options(grad_options))
        if include_hess:
            hess_graphs = list(list(g.diff(w) for w in wrt) for g in grad_graphs)
            hess = lambdify(inp, hess_graphs, **_get_lambidfy_options(hess_options))
    return BuildFunctionsResult(func=func, grad=grad, hess=hess)


@cacheit
def build_constraint_functions(variables, constraints, parameters=None, func_options=None, jac_options=None, hess_options=None):
    """Build callables functions for the constraints, constraint Jacobian, and constraint Hessian.

    Parameters
    ----------
    variables : List[sympy.core.symbol.Symbol]
        Free variables in the constraint expressions. By convention these are usually all
        StateVariables instances.
    constraints : List[sympy.core.expr.Expr]
        List of SymPy expression to compile
    parameters : Optional[List[sympy.core.symbol.Symbol]]
        Free variables in the sympy_graph. These are typically external
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

    """
    if parameters is None:
        parameters = []
    else:
        parameters = [wrap_symbol_symengine(p) for p in parameters]
    variables = tuple(variables)
    wrt = variables
    parameters = tuple(parameters)
    constraint_func, jacobian_func, hessian_func = None, None, None
    inp = sympify(variables + parameters)
    graph = sympify(constraints)
    constraint_func = lambdify(inp, [graph], **_get_lambidfy_options(func_options))

    grad_graphs = list(list(c.diff(w) for w in wrt) for c in graph)
    jacobian_func = lambdify(inp, grad_graphs, **_get_lambidfy_options(jac_options))

    hess_graphs = list(list(list(g.diff(w) for w in wrt) for g in c) for c in grad_graphs)
    hessian_func = lambdify(inp, hess_graphs, **_get_lambidfy_options(hess_options))
    return ConstraintFunctions(cons_func=constraint_func, cons_jac=jacobian_func, cons_hess=hessian_func)
