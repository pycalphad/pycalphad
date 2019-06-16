"""
This module constructs gradient functions for Models.
"""
from pycalphad.core.cache import cacheit
from pycalphad.core.utils import wrap_symbol_symengine
from symengine import sympify, lambdify
from collections import namedtuple


BuildFunctionsResult = namedtuple('BuildFunctionsResult', ['func', 'grad', 'hess'])


@cacheit
def build_functions(sympy_graph, variables, parameters=None, wrt=None, include_obj=True, include_grad=False, include_hess=False, cse=True):
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
    if include_obj:
        func = lambdify(inp, [graph], backend='lambda', cse=cse)
    if include_grad or include_hess:
        grad_graphs = list(graph.diff(w) for w in wrt)
        if include_grad:
            grad = lambdify(inp, grad_graphs, backend='lambda', cse=cse)
        if include_hess:
            hess_graphs = list(list(g.diff(w) for w in wrt) for g in grad_graphs)
            hess = lambdify(inp, hess_graphs, backend='lambda', cse=cse)
    return BuildFunctionsResult(func=func, grad=grad, hess=hess)
