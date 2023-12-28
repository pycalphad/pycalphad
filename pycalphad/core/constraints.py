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

def build_phase_local_constraints(mod, variables, phase_local_conditions, parameters=None):
    import pycalphad.variables as v
    phase_local_constraints = []
    for key, value in phase_local_conditions.items():
        # Should each phase-local condition key have a `.as_equation(model)` function?
        # That may work better as we expand to linear combinations of PLCs (fewer special cases needed)
        if isinstance(key, v.MoleFraction):
            cons = mod.moles(key.species, per_formula_unit=True) - \
                value * sum(mod.moles(v.Species(el), per_formula_unit=True) for el in mod.nonvacant_elements)
        else:
            cons = key - value
        phase_local_constraints.append(cons.expand())

    cf_output = build_constraint_functions(variables, phase_local_constraints,
                                           parameters=parameters)
    internal_cons_func = cf_output.cons_func
    internal_cons_jac = cf_output.cons_jac
    internal_cons_hess = cf_output.cons_hess

    return ConstraintTuple(internal_cons_func=internal_cons_func, internal_cons_jac=internal_cons_jac,
                           internal_cons_hess=internal_cons_hess,
                           num_internal_cons=len(phase_local_constraints))
