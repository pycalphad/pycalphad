from pycalphad import variables as v
from pycalphad.core.lower_convex_hull import lower_convex_hull
from pycalphad.core.light_dataset import LightDataset
from xarray import Dataset
import numpy as np
from collections import OrderedDict


def starting_point(conditions, state_variables, phase_records, grid, given_starting_point=None):
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
    given_starting_point : list of (phase name, dof) tuple
        Return the starting point given by the input.

    Returns
    -------
    LightDataset
    """
    from pycalphad import __version__ as pycalphad_version
    active_phases = sorted(phase_records.keys())
    # Ensure that '_FAKE_' will fit in the phase name array
    max_phase_name_len = max(max([len(x) for x in active_phases]), 6)
    maximum_internal_dof = max(prx.phase_dof for prx in phase_records.values())
    nonvacant_elements = phase_records[active_phases[0]].nonvacant_elements
    coord_dict = OrderedDict([(str(key), value) for key, value in conditions.items()])
    grid_shape = tuple(len(x) for x in coord_dict.values())
    max_phases = len(nonvacant_elements) + 1 # +1 is to accommodate the degenerate degree of freedom at the invariant reactions
    if given_starting_point is not None:
        max_phases = max(max_phases, len(given_starting_point))
    coord_dict['vertex'] = np.arange(max_phases)
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
    result = LightDataset(
        {'NP':     (conds_as_strings + ['vertex'], np.empty(grid_shape + (max_phases,))),
         'GM':     (conds_as_strings, np.empty(grid_shape)),
         'MU':     (conds_as_strings + ['component'], np.empty(grid_shape + (len(nonvacant_elements),))),
         'X':      (conds_as_strings + ['vertex', 'component'],
                    np.empty(grid_shape + (max_phases, len(nonvacant_elements),))),
         'Y':      (conds_as_strings + ['vertex', 'internal_dof'],
                    np.empty(grid_shape + (max_phases, maximum_internal_dof,))),
         'Phase':  (conds_as_strings + ['vertex'],
                    np.empty(grid_shape + (max_phases,), dtype='U%s' % max_phase_name_len)),
         'points': (conds_as_strings + ['vertex'],
                    np.empty(grid_shape + (max_phases,), dtype=np.int32))
         }, coords=coord_dict, attrs={'engine': 'pycalphad %s' % pycalphad_version})

    if given_starting_point is None:
        result = lower_convex_hull(grid, state_variables, result)
    else:
        out_energy = np.zeros(len(given_starting_point))
        out_moles = np.zeros((len(given_starting_point), 1, len(nonvacant_elements)))
        for phase_idx, (phase_name, phase_dof) in enumerate(given_starting_point):
            phase_dof_without_statevars = phase_dof[len(state_variables):]
            result['NP'][..., phase_idx] = 1./len(given_starting_point)
            phase_records[phase_name].obj(out_energy[phase_idx], np.atleast_2d(phase_dof))
            for comp_idx in range(len(nonvacant_elements)):
                phase_records[phase_name].mass_obj(out_moles[phase_idx], np.atleast_2d(phase_dof), comp_idx)
            result['Phase'][..., phase_idx] = phase_name
            result['Y'][..., phase_idx, :len(phase_dof_without_statevars)] = phase_dof_without_statevars
            result['Y'][..., phase_idx, len(phase_dof_without_statevars):] = np.nan
            out_energy[:] = 0
            out_moles[:, :] = 0
        result['X'][...] = out_moles[:, 0, :]
        result['GM'][...] = out_energy.mean()
        result['MU'][...] = out_energy.mean()
        result.remove('points')

    return result
