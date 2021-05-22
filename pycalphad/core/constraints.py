from pycalphad.core.constants import INTERNAL_CONSTRAINT_SCALING
from pycalphad.codegen.sympydiff_utils import build_constraint_functions
from collections import namedtuple

ConstraintTuple = namedtuple('ConstraintTuple', ['internal_cons_func', 'internal_cons_jac', 'internal_cons_hess',
                                                 'num_internal_cons'])


def build_constraints(mod, variables, parameters=None):
    internal_constraints = mod.get_internal_constraints()
    internal_constraints = [INTERNAL_CONSTRAINT_SCALING*x for x in internal_constraints]

    cf_output = build_constraint_functions(variables, internal_constraints,
                                           parameters=parameters)
    internal_cons_func = cf_output.cons_func
    internal_cons_jac = cf_output.cons_jac
    internal_cons_hess = cf_output.cons_hess

    return ConstraintTuple(internal_cons_func=internal_cons_func, internal_cons_jac=internal_cons_jac,
                           internal_cons_hess=internal_cons_hess,
                           num_internal_cons=len(internal_constraints))
