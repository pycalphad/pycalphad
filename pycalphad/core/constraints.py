from sympy import ImmutableMatrix, MatrixSymbol, Symbol
from pycalphad.codegen.sympydiff_utils import AutowrapFunction, CompileLock
from pycalphad.core.cache import cacheit
from pycalphad.core.constants import INTERNAL_CONSTRAINT_SCALING, MULTIPHASE_CONSTRAINT_SCALING
from pycalphad import variables as v
from collections import namedtuple


ConstraintFunctions = namedtuple('ConstraintFunctions', ['cons', 'jac', 'cons_hess'])


@cacheit
def _build_constraint_functions(variables, constraints, include_hess=False, parameters=None):
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
    wrt = variables
    params = MatrixSymbol('params', 1, len(parameters))
    inp_nobroadcast = MatrixSymbol('inp', 1, len(variables))
    args_nobroadcast = []
    for indx in range(len(variables)):
        args_nobroadcast.append(inp_nobroadcast[0, indx])
    for indx in range(len(parameters)):
        args_nobroadcast.append(params[0, indx])

    args = (inp_nobroadcast, params)
    nobroadcast = dict(zip(variables + parameters, args_nobroadcast))
    constraint_func = AutowrapFunction(args, ImmutableMatrix([c.xreplace(nobroadcast) for c in constraints]))

    jacobian = []
    hessian = []
    for constraint in constraints:
        sympy_graph_nobroadcast = constraint.xreplace(nobroadcast)
        with CompileLock:
            row = list(sympy_graph_nobroadcast.diff(nobroadcast[i]) for i in wrt)
        jacobian.append(row)
        if include_hess:
            col = list(x.diff(nobroadcast[i]) for i in wrt for x in row)
            hessian.append(col)
    jacobian_func = AutowrapFunction(args, ImmutableMatrix(jacobian))
    if len(hessian) > 0:
        hessian_func = AutowrapFunction(args, ImmutableMatrix(hessian))
    else:
        hessian_func = None
    return ConstraintFunctions(cons=constraint_func, jac=jacobian_func, cons_hess=hessian_func)


ConstraintTuple = namedtuple('ConstraintTuple', ['internal_cons', 'internal_jac', 'internal_cons_hess',
                                                 'multiphase_cons', 'multiphase_jac',
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
    # TODO: Conditions needing Hessians should probably have a 'second-order' tag or something
    has_chempots = any(str(cond).startswith('MU') for cond in conds.keys())
    cf_output = _build_constraint_functions(variables, internal_constraints,
                                            include_hess=has_chempots, parameters=parameters)
    internal_cons = cf_output.cons
    internal_jac = cf_output.jac
    internal_cons_hess = cf_output.cons_hess

    result_build = _build_constraint_functions(variables + [Symbol('NP')],
                                               multiphase_constraints, include_hess=False,
                                               parameters=parameters)
    multiphase_cons = result_build.cons
    multiphase_jac = result_build.jac
    return ConstraintTuple(internal_cons=internal_cons, internal_jac=internal_jac, internal_cons_hess=internal_cons_hess,
                           multiphase_cons=multiphase_cons, multiphase_jac=multiphase_jac,
                           num_internal_cons=len(internal_constraints), num_multiphase_cons=len(multiphase_constraints))


def get_multiphase_constraint_rhs(conds):
    return [MULTIPHASE_CONSTRAINT_SCALING*float(value) for cond, value in conds.items() if is_multiphase_constraint(cond)]
