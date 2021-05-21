from pycalphad import variables as v
from pycalphad.core.lower_convex_hull import lower_convex_hull
from pycalphad.core.light_dataset import LightDataset
from xarray import Dataset
import numpy as np
from collections import OrderedDict


def global_min_is_possible(conditions, state_variables):
    """
    Determine whether global minimization is possible
    to perform under the given set of conditions.
    Global minimization is possible when only T, P, N,
    compositions and/or chemical potentials are specified,
    but may not be possible with other conditions because
    there may be multiple (or zero) solutions.

    Parameters
    ----------
    conditions : dict
    state_variables : iterable of StateVariables

    Returns
    -------
    bool
    """
    global_min = True
    for cond in conditions.keys():
        if cond in state_variables or \
           isinstance(cond, v.MoleFraction) or \
           isinstance(cond, v.ChemicalPotential) or \
           cond == v.N:
            continue
        global_min = False
    return global_min


def starting_point(conditions, state_variables, phase_records, grid):
    """
    Find a starting point for the solution using a sample of the system energy surface.

    Parameters
    ----------
    conditions : OrderedDict
        Mapping of StateVariable to array of condition values.
    state_variables : list
        A list of the state variables (e.g., N, P, T) used in this calculation.
    phase_records : dict
        Mapping of phase names (strings) to PhaseRecords.
    grid : Dataset
        A sample of the energy surface of the system. The sample should at least
        cover the same state variable space as specified in the conditions.

    Returns
    -------
    Dataset
    """
    global_min_enabled = global_min_is_possible(conditions, state_variables)
    from pycalphad import __version__ as pycalphad_version
    active_phases = sorted(phase_records.keys())
    # Ensure that '_FAKE_' will fit in the phase name array
    max_phase_name_len = max(max([len(x) for x in active_phases]), 6)
    maximum_internal_dof = max(prx.phase_dof for prx in phase_records.values())
    nonvacant_elements = phase_records[active_phases[0]].nonvacant_elements
    coord_dict = OrderedDict([(str(key), value) for key, value in conditions.items()])
    grid_shape = tuple(len(x) for x in coord_dict.values())
    coord_dict['vertex'] = np.arange(
        len(nonvacant_elements) + 1)  # +1 is to accommodate the degenerate degree of freedom at the invariant reactions
    coord_dict['component'] = nonvacant_elements
    conds_as_strings = [str(k) for k in conditions.keys()]
    specified_elements = set()
    for i in conditions.keys():
        # Assume that a condition specifying a species contributes to constraining it
        if not hasattr(i, 'species'):
            continue
        specified_elements |= set(i.species.constituents.keys()) - {'VA'}
    dependent_comp = set(nonvacant_elements) - specified_elements
    if len(dependent_comp) != 1:
        raise ValueError('Number of dependent components different from one')

    ds_vars = {'NP':     (conds_as_strings + ['vertex'], np.empty(grid_shape + (len(nonvacant_elements)+1,))),
               'GM':     (conds_as_strings, np.empty(grid_shape)),
               'MU':     (conds_as_strings + ['component'], np.empty(grid_shape + (len(nonvacant_elements),))),
               'X':      (conds_as_strings + ['vertex', 'component'],
                          np.empty(grid_shape + (len(nonvacant_elements)+1, len(nonvacant_elements),))),
               'Y':      (conds_as_strings + ['vertex', 'internal_dof'],
                          np.empty(grid_shape + (len(nonvacant_elements)+1, maximum_internal_dof,))),
               'Phase':  (conds_as_strings + ['vertex'],
                          np.empty(grid_shape + (len(nonvacant_elements)+1,), dtype='U%s' % max_phase_name_len)),
               'points': (conds_as_strings + ['vertex'],
                          np.empty(grid_shape + (len(nonvacant_elements)+1,), dtype=np.int32))
               }

    # If we have free state variables, they will also be data variables / output variables
    free_statevars = sorted(set(state_variables) - set(conditions.keys()))
    for f_sv in free_statevars:
        ds_vars.update({str(f_sv): (conds_as_strings, np.empty(grid_shape))})

    result = LightDataset(ds_vars, coords=coord_dict, attrs={'engine': 'pycalphad %s' % pycalphad_version})
    if global_min_enabled:
        result = lower_convex_hull(grid, state_variables, result)
    else:
        raise NotImplementedError('Conditions not yet supported')

    return result
