from symengine import sympify, lambdify, Symbol
from pycalphad.core.cache import cacheit
from pycalphad import variables as v
from pycalphad.core.constants import INTERNAL_CONSTRAINT_SCALING, MULTIPHASE_CONSTRAINT_SCALING
from pycalphad.core.utils import wrap_symbol_symengine
from collections import namedtuple


ConstraintFunctions = namedtuple('ConstraintFunctions', ['cons_func', 'cons_jac', 'cons_hess'])


@cacheit
def _build_constraint_functions(variables, constraints, include_hess=False, parameters=None, cse=True):
    if parameters is None:
        parameters = []
    else:
        parameters = [wrap_symbol_symengine(p) for p in parameters]
    variables = tuple(variables)
    wrt = variables
    parameters = tuple(parameters)
    constraint__func, jacobian_func, hessian_func = None, None, None
    inp = sympify(variables + parameters)
    graph = sympify(constraints)
    constraint_func = lambdify(inp, [graph], backend='lambda', cse=cse)
    grad_graphs = list(list(c.diff(w) for w in wrt) for c in graph)
    jacobian_func = lambdify(inp, grad_graphs, backend='lambda', cse=cse)
    if include_hess:
        hess_graphs = list(list(list(g.diff(w) for w in wrt) for g in c) for c in grad_graphs)
        hessian_func = lambdify(inp, hess_graphs, backend='lambda', cse=cse)
    return ConstraintFunctions(cons_func=constraint_func, cons_jac=jacobian_func, cons_hess=hessian_func)


ConstraintTuple = namedtuple('ConstraintTuple', ['internal_cons', 'internal_jac', 'internal_cons_hess',
                                                 'multiphase_cons', 'multiphase_jac', 'multiphase_cons_hess',
                                                 'num_internal_cons', 'num_multiphase_cons'])


def is_multiphase_constraint(cond):
    cond = str(cond)
    if cond == 'N' or cond.startswith('X_'):
        return True
    else:
        return False


def build_constraints(mod, variables, conds, parameters=None):
    internal_constraints = mod.get_internal_constraints()
    internal_constraints = [INTERNAL_CONSTRAINT_SCALING*x for x in internal_constraints]
    multiphase_constraints = mod.get_multiphase_constraints(conds)
    multiphase_constraints = [MULTIPHASE_CONSTRAINT_SCALING*x for x in multiphase_constraints]
    cf_output = _build_constraint_functions(variables, internal_constraints,
                                            include_hess=True, parameters=parameters)
    internal_cons = cf_output.cons_func
    internal_jac = cf_output.cons_jac
    internal_cons_hess = cf_output.cons_hess

    result_build = _build_constraint_functions(variables + [Symbol('NP')],
                                               multiphase_constraints, include_hess=True,
                                               parameters=parameters)
    multiphase_cons = result_build.cons_func
    multiphase_jac = result_build.cons_jac
    multiphase_cons_hess = result_build.cons_hess
    return ConstraintTuple(internal_cons=internal_cons, internal_jac=internal_jac, internal_cons_hess=internal_cons_hess,
                           multiphase_cons=multiphase_cons, multiphase_jac=multiphase_jac, multiphase_cons_hess=multiphase_cons_hess,
                           num_internal_cons=len(internal_constraints), num_multiphase_cons=len(multiphase_constraints))


def get_multiphase_constraint_rhs(conds):
    return [MULTIPHASE_CONSTRAINT_SCALING*float(value) for cond, value in conds.items() if is_multiphase_constraint(cond)]
