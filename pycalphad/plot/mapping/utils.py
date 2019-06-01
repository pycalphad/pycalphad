from __future__ import print_function
import numpy as np
from collections import OrderedDict
from xarray import Dataset

from pycalphad import calculate, variables as v
from pycalphad.core.errors import ConditionError
from pycalphad.core.lower_convex_hull import lower_convex_hull
from pycalphad.core.equilibrium import _adjust_conditions
from pycalphad.core.cartesian import cartesian
from pycalphad.core.hyperplane import hyperplane
from pycalphad.core.constants import MIN_SITE_FRACTION
from .compsets import CompSet2D, CompSet


def build_composition_grid(components, conditions):
    """
    Create a cartesion grid of compositions, including adding the dependent component.

    Parameters
    ----------
    components : list of str
        List of component names
    conditions : dict
        Dictionary of pycalphad conditions

    Returns
    -------
    np.ndarray
        2D array of (M compositions, N components)

    """
    comp_conds = sorted([x for x in conditions.keys() if isinstance(x, v.X)])

    if len(comp_conds) > 0:
        comp_values = cartesian([conditions[cond] for cond in comp_conds])
        # Insert dependent composition value
        # TODO: Handle W(comp) as well as X(comp) here
        specified_components = {x.species.name for x in comp_conds}
        all_comps = set(components) - {'VA'}
        dependent_component = all_comps - specified_components
        dependent_component = list(dependent_component)
        if len(dependent_component) != 1:
            raise ValueError('Number of dependent components is different from one')
        else:
            dependent_component = dependent_component[0]
        insert_idx = sorted(all_comps).index(dependent_component)
        comp_values = np.concatenate((comp_values[..., :insert_idx],
                                      1 - np.sum(comp_values, keepdims=True, axis=-1),
                                      comp_values[..., insert_idx:]),
                                     axis=-1)
        # Prevent compositions near an edge from going negative
        comp_values[np.nonzero(comp_values < MIN_SITE_FRACTION)] = MIN_SITE_FRACTION*10
        # TODO: Assumes N=1
        comp_values /= comp_values.sum(axis=-1, keepdims=True)
    return comp_values


def convex_hull(dbf, comps, phases, conditions, model=None, calc_opts=None, parameters=None, callables=None):
    """
    1D convex hull for fixed potentials.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list or dict
        Names of phases to consider in the calculation.
    conditions : dict
        StateVariables and their corresponding value.
    model : Model, a dict of p  hase names to Model, or a seq of both, optional
        Model class to use for each phase.
    calc_opts : dict, optional
        Keyword arguments to pass to `calculate`, the energy/property calculation routine.
    parameters : dict, optional
        Maps SymPy Symbol to numbers, for overriding the values of parameters in the Database.
    callables : dict, optional
        Pre-computed callable functions for equilibrium calculation.

    Returns
    -------
    tuple
        Tuple of (Gibbs energies, phases, phase fractions, compositions, site fractions, chemical potentials)

    Notes
    -----
    Assumes that potentials are fixed and there is just a 1d composition grid.
    Minimizes the use of Dataset objects.
    """
    from pycalphad import __version__ as pycalphad_version
    calc_opts = calc_opts or {}
    conditions = _adjust_conditions(conditions)

    # 'calculate' accepts conditions through its keyword arguments
    if 'pdens' not in calc_opts:
        calc_opts['pdens'] = 2000
    grid = calculate(dbf, comps, phases, T=conditions[v.T], P=conditions[v.P],
                     parameters=parameters, fake_points=True, output='GM',
                     callables=callables, model=model, N=1, **calc_opts)


    active_phases = sorted(phases)
    # Ensure that '_FAKE_' will fit in the phase name array
    max_phase_name_len = max(max([len(x) for x in active_phases]), 6)
    from pycalphad.core.utils import generate_dof, unpack_components, get_state_variables, instantiate_models, get_pure_elements
    models = instantiate_models(dbf, comps, phases, model=model)
    active_comps = unpack_components(dbf, comps)
    maximum_internal_dof = 0
    for name, ph_obj in dbf.phases.items():
        dof = generate_dof(ph_obj, active_comps)
        maximum_internal_dof = max((len(dof[0]), maximum_internal_dof))

    state_variables = get_state_variables(models=models, conds=conditions)
    nonvacant_elements = get_pure_elements(dbf, comps)
    coord_dict = OrderedDict([(str(key), value) for key, value in conditions.items()])
    coord_dict.update({key: value/10 for key, value in coord_dict.items() if isinstance(key, v.X)})
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
    GM_values = result.GM.values.squeeze()
    simplex_phases = result.Phase.values.squeeze()
    phase_fractions = result.NP.values.squeeze()
    phase_compositions = result.X.values.squeeze()
    phase_site_fracs = result.Y.values.squeeze()
    chempots = result.MU.values.squeeze()
    return GM_values, simplex_phases, phase_fractions, phase_compositions, phase_site_fracs, chempots

def old_convex_hull(dbf, comps, phases, conditions, model=None,
                calc_opts=None, parameters=None, callables=None):
    """
    1D convex hull for fixed potentials.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list or dict
        Names of phases to consider in the calculation.
    conditions : dict
        StateVariables and their corresponding value.
    model : Model, a dict of phase names to Model, or a seq of both, optional
        Model class to use for each phase.
    calc_opts : dict, optional
        Keyword arguments to pass to `calculate`, the energy/property calculation routine.
    parameters : dict, optional
        Maps SymPy Symbol to numbers, for overriding the values of parameters in the Database.
    callables : dict, optional
        Pre-computed callable functions for equilibrium calculation.

    Returns
    -------
    tuple
        Tuple of (Gibbs energies, phases, phase fractions, compositions, site fractions, chemical potentials)

    Notes
    -----
    Assumes that potentials are fixed and there is just a 1d composition grid.
    Minimizes the use of Dataset objects.

    """
    calc_opts = calc_opts or {}
    conditions = _adjust_conditions(conditions)

    # 'calculate' accepts conditions through its keyword arguments
    if 'pdens' not in calc_opts:
        calc_opts['pdens'] = 2000
    grid = calculate(dbf, comps, phases, T=conditions[v.T], P=conditions[v.P],
                     parameters=parameters, fake_points=True, output='GM',
                     callables=callables, model=model, **calc_opts)

    # assume only one independent component
    indep_comp_conds = [c for c in conditions if isinstance(c, v.X)]
    num_indep_comp = len(indep_comp_conds)
    if num_indep_comp != 1:
        raise ConditionError(
            "Convex hull independent components different than one.")
    max_num_phases = (len(indep_comp_conds) + 1,)  # Gibbs phase rule
    comp_grid = build_composition_grid(comps, conditions)
    calc_grid_shape = comp_grid.shape[:-1]
    num_comps = comp_grid.shape[-1:]

    grid_energy_values = grid.GM.values.squeeze()
    grid_composition_values = grid.X.values.squeeze()
    grid_site_frac_values = grid.Y.values.squeeze()
    grid_phase_values = grid.Phase.values.squeeze()
    # construct the arrays to pass to hyperplane
    phase_fractions = np.empty(calc_grid_shape + max_num_phases)
    chempots = np.empty(calc_grid_shape + num_comps)
    simplex_points = np.empty(calc_grid_shape + max_num_phases, dtype=np.int32)

    comp_conds_idx = np.arange(0, num_comps[0], dtype=np.uint64)
    pot_conds_idx = np.array([], dtype=np.uint64)
    N = 1
    it = np.nditer(np.empty(calc_grid_shape), flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        hyperplane(grid_composition_values, grid_energy_values, comp_grid[idx],
                   chempots[idx], N, pot_conds_idx, comp_conds_idx,
                   phase_fractions[idx], simplex_points[idx])
        it.iternext()
    simplex_phases = grid_phase_values[simplex_points]  # shape: (calc_grid_shape + max_num_phases)
    GM_values = grid_energy_values[simplex_points]  # shape: (calc_grid_shape + max_num_phases)
    phase_compositions = grid_composition_values[simplex_points]  # shape: (calc_grid_shape + max_num_phases + num_comps)
    phase_site_fracs = grid_site_frac_values[simplex_points]  # shape: (calc_grid_shape + max_num_phases + num_internal_dof)

    return (GM_values, simplex_phases, phase_fractions, phase_compositions, phase_site_fracs, chempots)


def get_num_phases(eq_dataset):
    """Return the number of phases in equilibrium from an equilibrium dataset"""
    return int(np.sum(eq_dataset.Phase.values != '', axis=-1, dtype=np.int))


def get_compsets(eq_dataset, indep_comp=None, indep_comp_index=None):
    """
    Return a CompSet2D object if a pair of composition sets is found in an
    equilibrium dataset. Otherwise return None.

    Parameters
    ----------
    eq_dataset :
    indep_comp :
    indep_comp_index :

    Returns
    -------
    CompSet2D
    """
    if indep_comp is None:
        indep_comp = [c for c in eq_dataset.coords if 'X_' in c][0][2:]
    if indep_comp_index is None:
        indep_comp_index = eq_dataset.component.values.tolist().index(indep_comp)
    extracted_compsets = CompSet.from_dataset_vertices(eq_dataset, indep_comp, indep_comp_index, 3)
    if len(extracted_compsets) == 2:
        return CompSet2D(extracted_compsets)
    else:
        return None


def sort_x_by_y(x, y):
    """Sort a list of x in the order of sorting y"""
    return [xx for _, xx in sorted(zip(y, x), key=lambda pair: pair[0])]


def find_two_phase_region_compsets(hull_output, temperature, indep_comp, indep_comp_idx, discrepancy_tol=0.001, misc_gap_tol=0.1, minimum_composition=None):
    """
    From a dataset at constant T and P, return the composition sets for a two
    phase region or that have the smallest index composition coordinate

    Parameters
    ----------
    dataset : xr.Dataset
        Equilibrium-like from pycalphad that has a `Phase` Data variable.
    indep_comp_coord : str
        Coordinate name of the independent component

    Returns
    -------
    CompSet2D

    """
    phases, compositions, site_fracs = hull_output[1], hull_output[3], hull_output[4]
    grid_shape = phases.shape[:-1]
    num_phases = phases.shape[-1]
    it = np.nditer(np.empty(grid_shape), flags=['multi_index'])  # empty grid for indexing
    while not it.finished:
        idx = it.multi_index
        cs = []
        # TODO: assumption of only two phases, seems like the third phase index can have bad points
        # Three phases is probably an error anyways...
        if minimum_composition is not None and np.all(compositions[idx][:, indep_comp_idx][:2] < minimum_composition):
            it.iternext()
            continue
        for i in np.arange(num_phases):
            if str(phases[idx][i]) != '':
                stable_composition_sets = CompSet(str(phases[idx][i]), temperature, indep_comp, compositions[idx][i, indep_comp_idx], site_fracs[idx][i, :])
                cs.append(stable_composition_sets)
        if len(cs) == 2:
            compsets = CompSet2D(cs)
            if len(compsets.unique_phases) == 2:
                # we found a multiphase region, return them if the discrepancy is
                # above the tolerance
                if compsets.xdiscrepancy(ignore_phase=True) > discrepancy_tol:
                    return compsets
            else:
                # Same phase, either a single phase region or miscibility gap.
                if np.any(compsets.ydiscrepancy() > misc_gap_tol):
                    return compsets
        it.iternext()
    return None
