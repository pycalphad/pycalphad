"""
The lower_convex_hull module handles geometric calculations associated with
equilibrium calculation.
"""
from __future__ import print_function
from pycalphad.core.cartesian import cartesian
from pycalphad.core.constants import MIN_SITE_FRACTION
from .hyperplane import hyperplane
import numpy as np

# The energetic difference, in J/mol-atom, below which is considered 'zero'
DRIVING_FORCE_TOLERANCE = 1e-8


def lower_convex_hull(global_grid, result_array):
    """
    Find the simplices on the lower convex hull satisfying the specified
    conditions in the result array.

    Parameters
    ----------
    global_grid : Dataset
        A sample of the energy surface of the system.
    result_array : Dataset
        This object will be modified!
        Coordinates correspond to conditions axes.

    Returns
    -------
    None. Results are written to result_array.

    Notes
    -----
    This routine will not check if any simplex is degenerate.
    Degenerate simplices will manifest with duplicate or NaN indices.

    Examples
    --------
    None yet.
    """
    indep_conds = sorted([x for x in sorted(result_array.coords.keys()) if x in ['T', 'P']])
    comp_conds = sorted([x for x in sorted(result_array.coords.keys()) if x.startswith('X_')])
    pot_conds = sorted([x for x in sorted(result_array.coords.keys()) if x.startswith('MU_')])

    # Determine starting combinations of chemical potentials and compositions
    # TODO: Check Gibbs phase rule compliance

    if len(pot_conds) > 0:
        raise NotImplementedError('Chemical potential conditions are not yet supported')

    # FIRST CASE: Only composition conditions specified
    # We only need to compute the dependent composition value directly
    # Initialize trial points as lowest energy point in the system
    if (len(comp_conds) > 0) and (len(pot_conds) == 0):
        comp_values = cartesian([result_array.coords[cond] for cond in comp_conds])
        # Insert dependent composition value
        # TODO: Handle W(comp) as well as X(comp) here
        specified_components = {x[2:] for x in comp_conds}
        dependent_component = set(result_array.coords['component'].values) - specified_components
        dependent_component = list(dependent_component)
        if len(dependent_component) != 1:
            raise ValueError('Number of dependent components is different from one')
        insert_idx = sorted(result_array.coords['component'].values).index(dependent_component[0])
        comp_values = np.concatenate((comp_values[..., :insert_idx],
                                      1 - np.sum(comp_values, keepdims=True, axis=-1),
                                      comp_values[..., insert_idx:]),
                                     axis=-1)
        # Prevent compositions near an edge from going negative
        comp_values[np.nonzero(comp_values < MIN_SITE_FRACTION)] = MIN_SITE_FRACTION*10
        # TODO: Assumes N=1
        comp_values /= comp_values.sum(axis=-1, keepdims=True)
        #print(comp_values)

    # SECOND CASE: Only chemical potential conditions specified
    # TODO: Implementation of chemical potential

    # THIRD CASE: Mixture of composition and chemical potential conditions
    # TODO: Implementation of mixed conditions

    # factored out via profiling
    result_array_GM_values = result_array.GM.values
    result_array_points_values = result_array.points.values
    result_array_MU_values = result_array.MU.values
    result_array_NP_values = result_array.NP.values
    result_array_X_values = result_array.X.values
    result_array_Y_values = result_array.Y.values
    result_array_Phase_values = result_array.Phase.values
    global_grid_GM_values = global_grid.GM.values
    global_grid_X_values = global_grid.X.values

    it = np.nditer(result_array_GM_values, flags=['multi_index'])
    comp_coord_shape = tuple(len(result_array.coords[cond]) for cond in comp_conds)
    while not it.finished:
        indep_idx = it.multi_index[:len(indep_conds)]
        if len(comp_conds) > 0:
            comp_idx = np.ravel_multi_index(it.multi_index[len(indep_conds):], comp_coord_shape)
            idx_comp_values = comp_values[comp_idx]
        else:
            idx_comp_values = np.atleast_1d(1.)
        idx_global_grid_X_values = global_grid_X_values[indep_idx]
        idx_global_grid_GM_values = global_grid_GM_values[indep_idx]
        idx_result_array_MU_values = result_array_MU_values[it.multi_index]
        idx_result_array_NP_values = result_array_NP_values[it.multi_index]
        idx_result_array_GM_values = result_array_GM_values[it.multi_index]
        idx_result_array_points_values = result_array_points_values[it.multi_index]
        result_array_GM_values[it.multi_index] = \
            hyperplane(idx_global_grid_X_values, idx_global_grid_GM_values,
                       idx_comp_values, idx_result_array_MU_values,
                       idx_result_array_NP_values, idx_result_array_points_values)
        # Copy phase values out
        points = result_array_points_values[it.multi_index]
        result_array_Phase_values[it.multi_index] = global_grid.Phase.values[indep_idx].take(points, axis=0)
        result_array_X_values[it.multi_index] = global_grid.X.values[indep_idx].take(points, axis=0)
        result_array_Y_values[it.multi_index] = global_grid.Y.values[indep_idx].take(points, axis=0)
        # Special case: Sometimes fictitious points slip into the result
        # This can happen when we calculate stoichimetric phases by themselves
        if '_FAKE_' in result_array_Phase_values[it.multi_index]:
            # Chemical potentials are meaningless in this case
            idx_result_array_MU_values[...] = 0
            new_energy = 0.
            molesum = 0.
            for idx in range(len(result_array_Phase_values[it.multi_index])):
                midx = it.multi_index + (idx,)
                if result_array_Phase_values[midx] == '_FAKE_':
                    result_array_Phase_values[midx] = ''
                    result_array_X_values[midx] = np.nan
                    result_array_Y_values[midx] = np.nan
                    idx_result_array_NP_values[idx] = np.nan
                else:
                    new_energy += idx_result_array_NP_values[idx] * global_grid.GM.values[np.index_exp[indep_idx + (points[idx],)]]
                    molesum += idx_result_array_NP_values[idx]
            result_array_GM_values[it.multi_index] = new_energy / molesum
        it.iternext()
    del result_array['points']
    return result_array
