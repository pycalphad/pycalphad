"""
This module constructs gradient functions for Models.
"""
from pycalphad.core.cache import cacheit
from pycalphad.core.utils import wrap_symbol_symengine
from symengine import sympify, lambdify, count_ops
from collections import namedtuple


BuildFunctionsResult = namedtuple('BuildFunctionsResult', ['func', 'grad', 'hess'])

BACKEND_OPS_THRESHOLD = 50000


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
    if count_ops(graph) > BACKEND_OPS_THRESHOLD:
        backend = 'lambda'
    else:
        backend = 'llvm'
    # TODO: did not replace zoo with oo
    if include_obj:
        func = lambdify(inp, [graph], backend=backend, cse=cse)
    if include_grad or include_hess:
        grad_graphs = list(graph.diff(w) for w in wrt)
        grad_ops = sum(count_ops(x) for x in grad_graphs)
        if grad_ops > BACKEND_OPS_THRESHOLD:
            grad_backend = 'lambda'
        else:
            grad_backend = 'llvm'
        if include_grad:
            grad = lambdify(inp, grad_graphs, backend=grad_backend, cse=cse)
        if include_hess:
            hess_graphs = list(list(g.diff(w) for w in wrt) for g in grad_graphs)
            hess_ops = sum(sum(count_ops(xy) for xy in x) for x in hess_graphs)
            if hess_ops > BACKEND_OPS_THRESHOLD:
                hess_backend = 'lambda'
            else:
                hess_backend = 'llvm'
            print(f'Hess backend: {hess_backend} ({hess_ops} ops)')
            # Hessians are hard-coded to always use the lambda backend, for performance
            hess = lambdify(inp, hess_graphs, backend=hess_backend, cse=cse)
    return BuildFunctionsResult(func=func, grad=grad, hess=hess)
