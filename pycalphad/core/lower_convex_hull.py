"""
The lower_convex_hull module handles geometric calculations associated with
equilibrium calculation.
"""
from pycalphad.property_framework.computed_property import LinearCombination
from .hyperplane import hyperplane
from pycalphad.variables import ChemicalPotential, MassFraction, MoleFraction, IndependentPotential, SiteFraction, Moles
import numpy as np


def lower_convex_hull(global_grid, state_variables, conds_keys, phase_record_factory, result_array):
    """
    Find the simplices on the lower convex hull satisfying the specified
    conditions in the result array.

    Parameters
    ----------
    global_grid : Dataset
        A sample of the energy surface of the system.
    state_variables : List[v.StateVariable]
        A list of the state variables (e.g., P, T) used in this calculation.
    conds_keys : List
        A list of the keys of the conditions used in this calculation.
    phase_record_factory : PhaseRecordFactory
        PhaseRecordFactory object corresponding to this calculation.
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
    local_conds_keys = [c for c in conds_keys if getattr(c, 'phase_name', None) is not None]
    str_conds_keys = [str(c) for c in conds_keys]

    # factored out via profiling
    result_array_GM_values = result_array.GM
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

    while not it.finished:
        primary_index = it.multi_index
        grid_index = []
        # Relies on being ordered
        for lc in local_conds_keys:
            coord_idx = conds_keys.index(lc)
            grid_index.append(primary_index[coord_idx])
        for sv in state_variables:
            if sv in conds_keys:
                coord_idx = conds_keys.index(sv)
                grid_index.append(primary_index[coord_idx])
            else:
                # free state variable
                grid_index.append(0)
        grid_index = tuple(grid_index)

        idx_global_grid_X_values = global_grid_X_values[grid_index]
        idx_global_grid_GM_values = global_grid_GM_values[grid_index]
        idx_result_array_MU_values = result_array_MU_values[it.multi_index]
        idx_result_array_MU_values[:] = 0
        idx_fixed_lincomb_molefrac_coefs = []
        idx_fixed_lincomb_molefrac_rhs = []
        idx_fixed_chempot_indices = []
        # Extensive (amount) conditions are collected here and converted, after the loop,
        # into a simplex-closure row plus (k-1) scale-invariant ratio rows.
        idx_amount_coefs = []
        idx_amount_rhs = []

        for coord_idx, str_cond_key in enumerate(sorted(result_array.coords.keys())):
            try:
                cond_key = conds_keys[str_conds_keys.index(str_cond_key)]
            except ValueError:
                continue
            rhs = result_array.coords[str_cond_key][primary_index[coord_idx]]
            if isinstance(cond_key, IndependentPotential):
                # Already handled above in construction of grid_index
                continue
            if isinstance(cond_key, SiteFraction):
                # Already handled above in construction of grid_index
                continue
            elif isinstance(cond_key, ChemicalPotential):
                component_idx = result_array.coords['component'].index(str(cond_key.species))
                idx_fixed_chempot_indices.append(component_idx)
                idx_result_array_MU_values[component_idx] = rhs
            elif isinstance(cond_key, MassFraction):
                # wA = k -> (1-k)*MWA*xA - k*MWB*xB - k*MWC*xC = 0
                component_idx = result_array.coords['component'].index(str(cond_key.species))
                coef_vector = np.zeros(num_comps)
                coef_vector -= rhs
                coef_vector[component_idx] += 1
                # multiply coef_vector times a vector of molecular weights
                coef_vector = np.multiply(coef_vector, phase_record_factory.molar_masses)
                idx_fixed_lincomb_molefrac_coefs.append(coef_vector)
                idx_fixed_lincomb_molefrac_rhs.append(0.)
            elif isinstance(cond_key, MoleFraction):
                if cond_key.phase_name is not None:
                    # Phase-local condition already handled in construction of grid
                    continue
                component_idx = result_array.coords['component'].index(str(cond_key.species))
                coef_vector = np.zeros(num_comps)
                coef_vector[component_idx] = 1
                idx_fixed_lincomb_molefrac_coefs.append(coef_vector)
                idx_fixed_lincomb_molefrac_rhs.append(rhs)
            elif isinstance(cond_key, Moles):
                # Extensive (amount) condition. Total N (species is None) selects all
                # components; component N(i) selects one.
                cond_species = getattr(cond_key, 'species', None)
                coef_vector = np.zeros(num_comps)
                if cond_species is None:
                    coef_vector[:] = 1.0
                else:
                    coef_vector[result_array.coords['component'].index(str(cond_species))] = 1.0
                idx_amount_coefs.append(coef_vector)
                idx_amount_rhs.append(rhs)
            elif isinstance(cond_key, LinearCombination):
                coef_vector = np.zeros(num_comps)
                if cond_key.denominator == 1:
                    for symbol_idx, symbol in enumerate(cond_key.symbols):
                        if symbol != 1:
                            coef_idx = result_array.coords['component'].index(str(symbol.species))
                            coef_vector[coef_idx] = cond_key.coefs[symbol_idx]
                        else:
                            idx_fixed_lincomb_molefrac_rhs.append(rhs-cond_key.coefs[symbol_idx])
                else:
                    # This is a molar ratio
                    denominator_idx = cond_key.symbols.index(cond_key.denominator)
                    for symbol_idx, symbol in enumerate(cond_key.symbols):
                        if symbol_idx == denominator_idx:
                            coef_idx = result_array.coords['component'].index(str(symbol.species))
                            coef_vector[coef_idx] = cond_key.coefs[symbol_idx] - rhs
                        elif symbol != 1:
                            coef_idx = result_array.coords['component'].index(str(symbol.species))
                            coef_vector[coef_idx] = cond_key.coefs[symbol_idx]
                        else:
                            if cond_key.coefs[symbol_idx] != 0:
                                # Constant term for molar ratio should be zero
                                raise ValueError(f'Unsupported condition {cond_key}')
                            idx_fixed_lincomb_molefrac_rhs.append(-cond_key.coefs[symbol_idx])
                idx_fixed_lincomb_molefrac_coefs.append(coef_vector)
            else:
                raise ValueError(f'Unsupported condition {cond_key}')

        # Simplex closure: composition fractions sum to 1.
        idx_fixed_lincomb_molefrac_coefs.append(np.ones(num_comps))
        idx_fixed_lincomb_molefrac_rhs.append(1.0)
        # Extensive conditions contribute scale-invariant (ratio) constraints, not absolute
        # ones: for amount conditions m, r_0 * (c_m . x) - r_m * (c_0 . x) = 0. With k amount
        # conditions this is k-1 rows, so a fraction (X) and an amount (N) on the same component
        # never conflict, and the system stays square.
        for m in range(1, len(idx_amount_coefs)):
            ratio_row = idx_amount_rhs[0] * idx_amount_coefs[m] - idx_amount_rhs[m] * idx_amount_coefs[0]
            idx_fixed_lincomb_molefrac_coefs.append(ratio_row)
            idx_fixed_lincomb_molefrac_rhs.append(0.0)

        idx_fixed_lincomb_molefrac_coefs = np.atleast_2d(idx_fixed_lincomb_molefrac_coefs)
        idx_fixed_lincomb_molefrac_rhs = np.atleast_1d(idx_fixed_lincomb_molefrac_rhs)
        idx_fixed_chempot_indices = np.array(idx_fixed_chempot_indices, dtype=np.uintp)

        idx_result_array_NP_values = result_array_NP_values[it.multi_index]
        idx_result_array_points_values = result_array_points_values[it.multi_index]

        result_array_GM_values[it.multi_index] = \
            hyperplane(idx_global_grid_X_values, idx_global_grid_GM_values,
                       idx_result_array_MU_values, idx_fixed_chempot_indices, idx_fixed_lincomb_molefrac_coefs, idx_fixed_lincomb_molefrac_rhs,
                       idx_result_array_NP_values, idx_result_array_points_values)
        # Copy phase values out
        points = result_array_points_values[it.multi_index]
        result_array_Phase_values[it.multi_index][:num_comps] = global_grid_Phase_values[grid_index].take(points, axis=0)[:num_comps]
        result_array_X_values[it.multi_index][:num_comps] = global_grid_X_values[grid_index].take(points, axis=0)[:num_comps]
        result_array_Y_values[it.multi_index][:num_comps,:global_grid_Y_values.shape[-1]] = \
            global_grid_Y_values[grid_index].take(points, axis=0)[:num_comps]
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
                    new_energy += idx_result_array_NP_values[idx] * global_grid.GM[np.index_exp[grid_index + (points[idx],)]]
                    molesum += idx_result_array_NP_values[idx]
            if molesum != 0:
                result_array_GM_values[it.multi_index] = new_energy / molesum
        it.iternext()
    result_array.remove('points')
    return result_array
