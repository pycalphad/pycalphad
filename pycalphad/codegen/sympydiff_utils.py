"""
This module constructs gradient functions for Models.
"""
from pycalphad.core.cache import cacheit
from pycalphad.core.utils import wrap_symbol_symengine
from symengine import sympify, lambdify
from collections import namedtuple


BuildFunctionsResult = namedtuple('BuildFunctionsResult', ['func', 'grad', 'hess'])


@cacheit
def build_functions(sympy_graph, variables, parameters=None, wrt=None, include_grad=False, include_hess=False, func_options=None, grad_options=None, hess_options=None):
    """Build function, gradient, and Hessian callables of the sympy_graph

    Parameters
    ----------
    sympy_graph : sympy.core.expr.Expr
        SymPy expression to compile,
        :math:`f(x) : \mathbb{R}^{n} \\rightarrow \mathbb{R}`,
        which will be called by ``sympy_graph(variables+parameters)``
    variables : List[sympy.core.symbol.Symbol]
        Free variables in the sympy_graph, by convention these are usually all
        StateVariables instances.
    parameters : Optional[List[sympy.core.symbol.Symbol]]
        Free variables in the sympy_graph, these are typically external
        parameters that are controlled by the user.
    wrt : Optional[List[sympy.core.symbol.Symbol]]
        Variables to differentiate *with respect to* for gradient and Hessian
        callables. If None, the default is to differentiate w.r.t. all variables.
    include_grad : Optional[bool]
        Whether to build the gradient callable,
        :math:`\\pmb{g} = \\nabla f(x) : \mathbb{R}^{n} \\rightarrow \mathbb{R}^{n}`
    include_hess : Optional[bool]
        Whether to build the Hessian callable,
        :math:`\mathbb{H} = \\nabla^2 f(x) : \mathbb{R}^{n} \\rightarrow \mathbb{R}^{n \\times n}`
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
    Default options for compiling the function, gradient and Hessian are::

        {'backend': 'llvm', 'cse': True}

    """
    def _get_options(user_options):
        default_options = {'backend': 'llvm', 'cse': True}
        if user_options is not None:
            default_options.update(user_options)
        return default_options

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
    func = lambdify(inp, [graph], **_get_options(func_options))
    if include_grad or include_hess:
        grad_graphs = list(graph.diff(w) for w in wrt)
        if include_grad:
            grad = lambdify(inp, grad_graphs, **_get_options(grad_options))
        if include_hess:
            hess_graphs = list(list(g.diff(w) for w in wrt) for g in grad_graphs)
            hess = lambdify(inp, hess_graphs, **_get_options(hess_options))
    return BuildFunctionsResult(func=func, grad=grad, hess=hess)
