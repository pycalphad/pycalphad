from typing import Union

import numpy as np

from pycalphad import variables as v
from pycalphad.core.composition_set import CompositionSet

from pycalphad.mapping.primitives import Point, Node

def degrees_of_freedom(point: Union[Point, Node], components: list[str], num_potential_conditions: int):
    """
    Degrees of freedom of point defined by Gibbs phase rule

    Components + 2 = Phases + DOF
        2 is offsetted by the number of potential conditions

    Parameters
    ----------
    point : Point or Node
    components : [str]
        List of components, VA will be ignored
    num_potential_conditions : number of variable potential conditions

    Returns
    -------
    int - degrees of freedom, 0 represents an invariant point/node
    """
    non_va_comps = len(set(components) - {"VA"})
    return non_va_comps + 2 - len(point.stable_composition_sets) - (2 - num_potential_conditions)

def is_state_variable(var: v.StateVariable):
    """
    Returns whether var is a state variable or not
    """
    return isinstance(var, (v.IndependentPotential, v.SystemMolesType))

def update_cs_phase_frac(comp_set: CompositionSet, phase_frac: float):
    """
    Wrapper to update the phase fraction of a composition set
    This just helps with splitting the cs.dof to state variables and constituents when updating

    Parameters
    ----------
    comp_set : CompositionSet
    phase_frac : float
    """
    num_sv = comp_set.phase_record.num_statevars
    comp_set.update(comp_set.dof[num_sv:], phase_frac, comp_set.dof[:num_sv])

def get_statevars_array(conditions: dict[v.StateVariable, float], state_variables: list[v.StateVariable] = None):
    """
    Creates numpy array of state variables in conditions

    Parameters
    ----------
    conditions : dict[v.StateVariable, float]

    Returns
    -------
    numpy array of len(state variables)
    """
    # If no state variables is given, then grab them from conditions
    if state_variables is None:
        state_variables = sorted([key for key in conditions if is_state_variable(key)], key=str)
    return np.asarray([conditions[sv] for sv in state_variables], dtype=np.float64)

def _sort_axis_by_state_vars(axis_vars: list[v.StateVariable]):
    """
    Sorts list of axis variables by [state variables (N, P, T)] + [non-state variables]

    Parameters
    ----------
    axis_vars : [v.StateVariable]

    Returns
    -------
    sorted axis vars : [v.StateVariable]
    """
    state_vars = [av for av in axis_vars if is_state_variable(av)]
    non_state_vars = [av for av in axis_vars if not is_state_variable(av)]
    return state_vars + non_state_vars

def _generate_point_with_fixed_cs(point: Point, cs_to_fix: CompositionSet, cs_to_free: CompositionSet):
    """
    Generates point with two cs, one fixed and one free

    Parameters
    ----------
    point : Point
        Point to generate new point from
    cs_to_fix : CompositionSet
    cs_to_free : CompositionSet

    Returns
    -------
    new_point : Point
        Retains same equilibria as previous point
        Conditions will be updated with the updated phase fractions of the composition sets
    """
    new_point = Point.with_copy(point.global_conditions, point.chemical_potentials, [cs_to_fix], [cs_to_free])

    new_point._fixed_composition_sets[0].fixed = True
    update_cs_phase_frac(new_point._fixed_composition_sets[0], 0.0)
    new_point._free_composition_sets[0].fixed = False
    update_cs_phase_frac(new_point._free_composition_sets[0], 1.0)

    for key in new_point.global_conditions:
        new_point.global_conditions[key] = new_point.get_property(key)
    return new_point

def _generate_point_with_free_cs(point: Point, bias_towards_free : bool = False):
    """
    Frees all composition sets in point and sets NP to 1/n for all CS
        If bias_towards_free, then all free cs in point will have a weight of 2 when normalizing phase fractions

    Parameters
    ----------
    point : Point
        Point to generate new point from
    bias_towards_free : bool
        Whether to bias composition towards free phases of previous point

    Returns
    -------
    new_point : Point
        Point will all composition sets free
    """
    phase_sum = 0
    for cs in point.stable_composition_sets:
        if bias_towards_free:
            phase_sum += 1 if cs.fixed else 2
        else:
            phase_sum += 1

    new_point = Point.with_copy(point.global_conditions, point.chemical_potentials, [], point.stable_composition_sets)
    for cs in new_point._free_composition_sets:
        if bias_towards_free:
            new_amt = 1 if cs.fixed else 2
            update_cs_phase_frac(cs, new_amt/phase_sum)
        else:
            update_cs_phase_frac(cs, 1/phase_sum)
        cs.fixed = False

    for key in new_point.global_conditions:
        new_point.global_conditions[key] = new_point.get_property(key)
    return new_point