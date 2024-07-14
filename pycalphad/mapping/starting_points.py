import logging
from typing import Union
import copy

import numpy as np

from pycalphad import Database, Workspace, as_property, variables as v
from pycalphad.core.composition_set import CompositionSet

from pycalphad.mapping.primitives import Point
import pycalphad.mapping.utils as map_utils

_log = logging.getLogger(__name__)

def point_from_equilibrium(dbf: Database, components: list[str], phases: list[str], conditions: dict[v.StateVariable, float], **eq_kwargs):
    """
    Converts Workspace equilibrium result to a Point

    Parameters
    ----------
    dbf : Database
    components : [str]
    phases : [str]
    conditions : dict[v.StateVariable, float]
    eq_kwargs : dict
        Additional arguments for Workspace (models, phase records, etc.)
    """
    wks = Workspace(dbf, components, phases, conditions, **eq_kwargs)

    chemical_potentials = np.squeeze(wks.eq.MU)
    comp_sets = wks.get_composition_sets()

    if len(comp_sets) == 0:
        return None

    copy_conds = copy.deepcopy(conditions)

    np_sum = sum([cs.NP for cs in comp_sets])
    for cs in comp_sets:
        map_utils.update_cs_phase_frac(cs, cs.NP/np_sum)

    for key in copy_conds:
        copy_conds[key] = key.compute_property(comp_sets, copy_conds, chemical_potentials)

    point = Point(copy_conds, chemical_potentials, [cs for cs in comp_sets if cs.fixed], [cs for cs in comp_sets if not cs.fixed])
    return point