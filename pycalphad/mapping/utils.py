from typing import Union

import numpy as np

from pycalphad import variables as v
from pycalphad.core.composition_set import CompositionSet

from pycalphad.mapping.primitives import Point, Node, STATEVARS

def degrees_of_freedom(point: Union[Point, Node], components: list[str], num_potential_conditions: int):
    non_va_comps = len(set(components) - {"VA"})
    return non_va_comps + 2 - len(point.stable_composition_sets) - (2 - num_potential_conditions)

def update_cs_phase_frac(comp_set: CompositionSet, phase_frac: float):
    comp_set.update(comp_set.dof[len(STATEVARS):], phase_frac, comp_set.dof[:len(STATEVARS)])

def get_statevars_array(conditions: dict[v.StateVariable, float]):
    return np.asarray([conditions[sv] for sv in STATEVARS], dtype=np.float64)

def elements_from_components(components: list[str]):
    # Components can be a compound, so we want to get all unique elements
    # excluding VA. We'll also sort them
    non_va_comps = list(set(components) - {"VA"})
    species_list = [v.Species(c) for c in non_va_comps]
    elements = set.union(*[set([key for key in s.constituents.keys()]) for s in species_list])
    return sorted(list(elements))

def _sort_axis_by_state_vars(axis_vars: list[v.StateVariable]):
    state_vars = [av for av in axis_vars if av in STATEVARS]
    non_state_vars = [av for av in axis_vars if av not in STATEVARS]
    return state_vars + non_state_vars

def _generate_point_with_fixed_cs(point: Point, cs_to_fix: CompositionSet, cs_to_free: CompositionSet):
    new_point = Point.with_copy(point.global_conditions, point.chemical_potentials, [cs_to_fix], [cs_to_free])

    new_point._fixed_composition_sets[0].fixed = True
    update_cs_phase_frac(new_point._fixed_composition_sets[0], 0.0)
    new_point._free_composition_sets[0].fixed = False
    update_cs_phase_frac(new_point._free_composition_sets[0], 1.0)

    for key in new_point.global_conditions:
        new_point.global_conditions[key] = new_point.get_property(key)
    return new_point

def _generate_point_with_free_cs(point: Point, bias_towards_free = False):
    """
    Frees all composition sets in point and sets NP to 1/n for all CS
        If bias_towards_free, then all free cs in point will have a weight of 2 when normalizing phase fractions
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

        # cs.fixed = False
        # update_cs_phase_frac(cs, 1/len(new_point._free_composition_sets))
    for key in new_point.global_conditions:
        new_point.global_conditions[key] = new_point.get_property(key)
    return new_point