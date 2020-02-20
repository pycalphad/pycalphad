from symengine import Symbol
from pycalphad.core.constants import INTERNAL_CONSTRAINT_SCALING, MULTIPHASE_CONSTRAINT_SCALING
from pycalphad.codegen.sympydiff_utils import build_constraint_functions
from collections import namedtuple

ConstraintTuple = namedtuple('ConstraintTuple', ['internal_cons_func', 'internal_cons_jac', 'internal_cons_hess',
                                                 'multiphase_cons_func', 'multiphase_cons_jac', 'multiphase_cons_hess',
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
    cf_output = build_constraint_functions(variables, internal_constraints,
                                           parameters=parameters)
    internal_cons_func = cf_output.cons_func
    internal_cons_jac = cf_output.cons_jac
    internal_cons_hess = cf_output.cons_hess

    result_build = build_constraint_functions(variables + [Symbol('NP')],
                                              multiphase_constraints,
                                              parameters=parameters)
    multiphase_cons_func = result_build.cons_func
    multiphase_cons_jac = result_build.cons_jac
    multiphase_cons_hess = result_build.cons_hess
    return ConstraintTuple(internal_cons_func=internal_cons_func, internal_cons_jac=internal_cons_jac,
                           internal_cons_hess=internal_cons_hess,
                           multiphase_cons_func=multiphase_cons_func, multiphase_cons_jac=multiphase_cons_jac,
                           multiphase_cons_hess=multiphase_cons_hess,
                           num_internal_cons=len(internal_constraints), num_multiphase_cons=len(multiphase_constraints))


def get_multiphase_constraint_rhs(conds):
    return [MULTIPHASE_CONSTRAINT_SCALING*float(value) for cond, value in conds.items() if is_multiphase_constraint(cond)]
