from pycalphad import Database, Model, variables as v
from pycalphad.core.solver import Solver
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.utils import instantiate_models
from pycalphad.codegen.sympydiff_utils import build_functions
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.minimizer import state_variable_differential, site_fraction_differential
import numpy as np


def dot_derivative(spec, state, property_records):
    """
    Sample the internal degrees of freedom of a phase.

    Parameters
    ----------
    spec : SystemSpecifications
        some description
    state : Title
        another description

    Returns
    -------
    dot derivative of property
    """
    property_of_interest = 'HM'
    statevar_of_interest = v.T
    state_variables = state.compsets[0].phase_record.state_variables
    statevar_idx = sorted(state_variables, key=str).index(statevar_of_interest)
    delta_chemical_potentials, delta_statevars, delta_phase_amounts = \
    state_variable_differential(spec, state, statevar_idx)

    
    # Sundman et al, 2015, Eq. 73
    dot_derivative = 0.0
    naive_derivative = 0.0
    for idx, compset in enumerate(state.compsets):
        phase_name = compset.phase_record.phase_name
        proprecord = property_records[phase_name]
        func_value, grad_value = proprecord.func(compset.dof), proprecord.grad(compset.dof)
        delta_sitefracs = site_fraction_differential(state.cs_states[idx], delta_chemical_potentials,
                                                     delta_statevars)
        
        dot_derivative += delta_phase_amounts[idx] * func_value
        dot_derivative += compset.NP * grad_value[statevar_idx] * delta_statevars[statevar_idx]
        naive_derivative += compset.NP * grad_value[statevar_idx] * delta_statevars[statevar_idx]
        dot_derivative += compset.NP * np.dot(delta_sitefracs, grad_value[len(state_variables):])


    return dot_derivative