"""
The lower_convex_hull module handles geometric calculations associated with
equilibrium calculation.
"""
from pycalphad.core.cartesian import cartesian
from pycalphad.core.constants import MIN_SITE_FRACTION
from .hyperplane import hyperplane
import numpy as np
import itertools


def lower_convex_hull(global_grid, state_variables, result_array):
    """
    Find the simplices on the lower convex hull satisfying the specified
    conditions in the result array.

    Parameters
    ----------
    global_grid : Dataset
        A sample of the energy surface of the system.
    state_variables : List[v.StateVariable]
        A list of the state variables (e.g., P, T) used in this calculation.
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
    state_variables = sorted(state_variables, key=str)
    comp_conds = sorted([x for x in sorted(result_array.coords.keys()) if x.startswith('X_')])
    comp_conds_indices = sorted([idx for idx, x in enumerate(sorted(result_array.coords['component']))
                                 if 'X_'+x in comp_conds])
    comp_conds_indices = np.array(comp_conds_indices, dtype=np.uintp)
    pot_conds = sorted([x for x in sorted(result_array.coords.keys()) if x.startswith('MU_')])
    pot_conds_indices = sorted([idx for idx, x in enumerate(sorted(result_array.coords['component']))
                                if 'MU_'+x in pot_conds])
    pot_conds_indices = np.array(pot_conds_indices, dtype=np.uintp)

    if len(set(pot_conds_indices) & set(comp_conds_indices)) > 0:
        raise ValueError('Cannot specify component chemical potential and amount simultaneously')

    if len(comp_conds) > 0:
        cart_values = cartesian([result_array.coords[cond] for cond in comp_conds])
    else:
        cart_values = np.atleast_2d(1.)
    # TODO: Handle W(comp) as well as X(comp) here
    comp_values = np.zeros(cart_values.shape[:-1] + (len(result_array.coords['component']),))
    for idx in range(comp_values.shape[-1]):
        if idx in comp_conds_indices:
            comp_values[..., idx] = cart_values[..., np.where(comp_conds_indices == idx)[0][0]]
        elif idx in pot_conds_indices:
            # Composition value not used
            comp_values[..., idx] = 0
        else:
            # Dependent component (composition value not used)
            comp_values[..., idx] = 0
    # Prevent compositions near an edge from going negative
    comp_values[np.nonzero(comp_values < MIN_SITE_FRACTION)] = MIN_SITE_FRACTION*10

    if len(pot_conds) > 0:
        cart_pot_values = cartesian([result_array.coords[cond] for cond in pot_conds])

    #result_array['Phase'] = force_indep_align(result_array.Phase)
    # factored out via profiling
    result_array_GM_values = result_array.GM
    result_array_GM_dims = result_array.data_vars['GM'][0]
    result_array_points_values = result_array.points
    result_array_MU_values = result_array.MU
    result_array_NP_values = result_array.NP
    result_array_X_values = result_array.X
    result_array_Y_values = result_array.Y
    result_array_Phase_values = result_array.Phase
    global_grid_GM_values = global_grid.GM
    global_grid_X_values = global_grid.X
    global_grid_Y_values = global_grid.Y
    global_grid_Phase_values = global_grid.Phase
    num_comps = len(result_array.coords['component'])

    it = np.nditer(result_array_GM_values, flags=['multi_index'])
    comp_coord_shape = tuple(len(result_array.coords[cond]) for cond in comp_conds)
    pot_coord_shape = tuple(len(result_array.coords[cond]) for cond in pot_conds)
    while not it.finished:
        indep_idx = []
        # Relies on being ordered
        for sv in state_variables:
            if str(sv) in result_array.coords.keys():
                coord_idx = list(result_array.coords.keys()).index(str(sv))
                indep_idx.append(it.multi_index[coord_idx])
            else:
                # free state variable
                indep_idx.append(0)
        indep_idx = tuple(indep_idx)
        if len(comp_conds) > 0:
            comp_idx = np.ravel_multi_index(tuple(idx for idx, key in zip(it.multi_index, result_array_GM_dims) if key in comp_conds), comp_coord_shape)
            idx_comp_values = comp_values[comp_idx, :]
        else:
            idx_comp_values = np.atleast_1d(1.)
        if len(pot_conds) > 0:
            pot_idx = np.ravel_multi_index(tuple(idx for idx, key in zip(it.multi_index, result_array_GM_dims) if key in pot_conds), pot_coord_shape)
            idx_pot_values = np.array(cart_pot_values[pot_idx, :])

        idx_global_grid_X_values = global_grid_X_values[indep_idx]
        idx_global_grid_GM_values = global_grid_GM_values[indep_idx]
        idx_result_array_MU_values = result_array_MU_values[it.multi_index]
        idx_result_array_MU_values[:] = 0
        for idx in range(len(pot_conds_indices)):
            idx_result_array_MU_values[pot_conds_indices[idx]] = idx_pot_values[idx]
        idx_result_array_NP_values = result_array_NP_values[it.multi_index]
        idx_result_array_points_values = result_array_points_values[it.multi_index]
        result_array_GM_values[it.multi_index] = \
            hyperplane(idx_global_grid_X_values, idx_global_grid_GM_values,
                       idx_comp_values, idx_result_array_MU_values, float(global_grid.coords['N'][0]),
                       pot_conds_indices, comp_conds_indices,
                       idx_result_array_NP_values, idx_result_array_points_values)
        # Copy phase values out
        points = result_array_points_values[it.multi_index]
        result_array_Phase_values[it.multi_index][:num_comps] = global_grid_Phase_values[indep_idx].take(points, axis=0)[:num_comps]
        result_array_X_values[it.multi_index][:num_comps] = global_grid_X_values[indep_idx].take(points, axis=0)[:num_comps]
        result_array_Y_values[it.multi_index][:num_comps] = global_grid_Y_values[indep_idx].take(points, axis=0)[:num_comps]
        # Special case: Sometimes fictitious points slip into the result
        if '_FAKE_' in result_array_Phase_values[it.multi_index]:
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
                    new_energy += idx_result_array_NP_values[idx] * global_grid.GM[np.index_exp[indep_idx + (points[idx],)]]
                    molesum += idx_result_array_NP_values[idx]
            result_array_GM_values[it.multi_index] = new_energy / molesum
        it.iternext()
    result_array.remove('points')
    return result_array
