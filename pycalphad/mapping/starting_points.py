import logging
from typing import Union
import copy

import numpy as np

from pycalphad import Database, Workspace, as_property, variables as v
from pycalphad.core.composition_set import CompositionSet

from pycalphad.mapping.primitives import Point
import pycalphad.mapping.utils as map_utils

_log = logging.getLogger(__name__)

def _sort_cs_to_fix(comp_sets: list[CompositionSet], conditions: dict[v.StateVariable, float], chemical_potentials: list[float]):
    """
    1. DOO
    2. Composition product
    """
    doo, prod = [], []
    for cs in comp_sets:
        doo.append(as_property(f"degree_of_ordering({cs.phase_record.phase_name})").compute_property([cs], conditions, chemical_potentials))
        if doo[-1] < 1e-3:
            doo[-1] = 0
        prod.append(np.prod(cs.X))
    
    for i in range(len(comp_sets)):
        for j in range(i+1, len(comp_sets)):
            if (doo[j] > doo[i]) or (abs(doo[j] - doo[i]) < 1e-3 and prod[j] < prod[i]):
                comp_sets[i], comp_sets[j] = comp_sets[j], comp_sets[i]

def point_from_equilibrium(dbf: Database, components: list[str], phases: list[str], conditions: dict[v.StateVariable, float], free_var: Union[v.StateVariable, list[v.StateVariable]] = None, **eq_kwargs):
    """
    1. Computes equilibrium with dbf, components, phases and conditions
    2. Free selected variables (if any), fixing composition sets by some heuristic on which CS should be fixed first (generally refers to CS with least degrees of freedom)
    3. Create point from composition set
    """
    wks = Workspace(dbf, components, phases, conditions, **eq_kwargs)

    chemical_potentials = np.squeeze(wks.eq.MU)
    for _, cs in wks.enumerate_composition_sets():
        comp_sets = cs

    if len(comp_sets) == 0:
        return None

    _sort_cs_to_fix(comp_sets, conditions, chemical_potentials)

    copy_conds = copy.deepcopy(conditions)

    if free_var is not None:
        if not hasattr(free_var, "len"):
            free_var = [free_var]

        for i in range(len(free_var)):
            comp_sets[i].fixed = True
            map_utils.update_cs_phase_frac(comp_sets[i], 0.0)
            del copy_conds[free_var[i]]

    np_sum = sum([cs.NP for cs in comp_sets])
    for cs in comp_sets:
        map_utils.update_cs_phase_frac(cs, cs.NP/np_sum)

    for key in copy_conds:
        copy_conds[key] = key.compute_property(comp_sets, copy_conds, chemical_potentials)

    point = Point(copy_conds, chemical_potentials, [cs for cs in comp_sets if cs.fixed], [cs for cs in comp_sets if not cs.fixed])
    return point

def create_point_with_free_variable(point: Point, free_var: Union[v.StateVariable, list[v.StateVariable]] = None):
    """
    Creates new point, freeing up variables and fixing composition sets of previous point

    Note: this currently assumes that all composition sets in point are free
    """
    comp_sets = point.stable_composition_sets

    _sort_cs_to_fix(comp_sets, point.global_conditions, point.chemical_potentials)
    copy_conds = copy.deepcopy(point.global_conditions)

    if free_var is not None:
        if not hasattr(free_var, "len"):
            free_var = [free_var]

        for i in range(len(free_var)):
            comp_sets[i].fixed = True
            map_utils.update_cs_phase_frac(comp_sets[i], 0.0)
            del copy_conds[free_var[i]]

    np_sum = sum([cs.NP for cs in comp_sets])
    for cs in comp_sets:
        map_utils.update_cs_phase_frac(cs, cs.NP/np_sum)

    for key in copy_conds:
        copy_conds[key] = key.compute_property(comp_sets, copy_conds, point.chemical_potentials)

    new_point = Point(copy_conds, point.chemical_potentials, [cs for cs in comp_sets if cs.fixed], [cs for cs in comp_sets if not cs.fixed])
    return new_point
        



    