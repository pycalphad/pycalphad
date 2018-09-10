from pycalphad import variables as v
from pycalphad.core.lower_convex_hull import lower_convex_hull
from xarray import Dataset
import numpy as np
from collections import namedtuple, OrderedDict

ConditionsResult = namedtuple('ConditionsResult', ['global_min'])


def analyze_conditions(conditions, state_variables):
    global_min = True
    for cond in conditions.keys():
        if cond in state_variables or \
           isinstance(cond, v.Composition) or \
           isinstance(cond, v.ChemicalPotential) or \
           cond == v.N:
            continue
        global_min = False
    return ConditionsResult(global_min=global_min)


def starting_point(conditions, state_variables, phase_records, grid):
    cond_analysis = analyze_conditions(conditions, state_variables)
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
    if cond_analysis.global_min:
        result = Dataset({'NP':     (conds_as_strings + ['vertex'], np.empty(grid_shape + (len(nonvacant_elements)+1,))),
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
                          },
                         coords=coord_dict, attrs={'engine': 'pycalphad %s' % pycalphad_version})
        result = lower_convex_hull(grid, state_variables, result)
    else:
        raise NotImplementedError('Conditions not yet supported')

    return result
