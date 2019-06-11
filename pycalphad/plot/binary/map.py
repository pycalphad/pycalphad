
import time
from copy import deepcopy
import numpy as np
from pycalphad import variables as v
from pycalphad.core.starting_point import starting_point
from pycalphad.core.eqsolver import _solve_eq_at_conditions
from pycalphad.core.utils import unpack_condition, extract_parameters, instantiate_models, unpack_components, get_state_variables
from pycalphad.codegen.callables import build_callables, build_phase_records
from .convex_hull import convex_hull
from .compsets import get_compsets, find_two_phase_region_compsets
from .zpf_boundary_sets import ZPFBoundarySets


def map_binary(dbf, comps, phases, conds, eq_kwargs=None, boundary_sets=None,
               verbose=False, summary=False,):
    """
    Map a binary T-X phase diagram

    Parameters
    ----------
    dbf : Database
    comps : list of str
    phases : list of str
        List of phases to consider in mapping
    conds : dict
        Dictionary of conditions
    eq_kwargs : dict
        Dictionary of keyword arguments to pass to equilibrium
    verbose : bool
        Print verbose output for mapping
    boundary_sets : ZPFBoundarySets
        Existing ZPFBoundarySets

    Returns
    -------
    ZPFBoundarySets

    Notes
    -----
    Assumes conditions in T and X.

    Simple algorithm to map a binary phase diagram in T-X. More or less follows
    the algorithm described in Figure 2 by Snider et al. [1] with the small
    algorithmic improvement of constructing a convex hull to find the next
    potential two phase region.

    For each temperature, proceed along increasing composition, skipping two
    over two phase regions, once calculated.

    [1] J. Snider, I. Griva, X. Sun, M. Emelianenko, Set based framework for
        Gibbs energy minimization, Calphad. 48 (2015) 18–26.
        doi:10.1016/j.calphad.2014.09.005.
    """
    eq_kwargs = eq_kwargs or {}
    # implictly add v.N to conditions
    if v.N not in conds:
        conds[v.N] = [1.0]

    species = unpack_components(dbf, comps)
    params = eq_kwargs.get('parameters', {})
    syms = sorted(extract_parameters(params)[0], key=str)
    models = eq_kwargs.get('model')
    statevars = get_state_variables(models=models, conds=conds)
    if models is None:
        # TODO: case fail if model is not a dict of instantiated models, e.g. Model class
        models = instantiate_models(dbf, comps, phases, model=eq_kwargs.get('model'),
                                    parameters=params, symbols_only=True)
        eq_kwargs['model'] = models
    if 'callables' not in eq_kwargs:
        cbs = build_callables(dbf, comps, phases, models, parameter_symbols=syms,
                              output='GM', additional_statevars={v.P, v.T, v.N})
        eq_kwargs['callables'] = cbs

    prxs = build_phase_records(dbf, species, phases, conds, models, output='GM',
                               callables=cbs, parameters=params,
                               build_gradients=True)

    indep_comp = [key for key, value in conds.items() if isinstance(key, v.Composition) and len(np.atleast_1d(value)) > 1]
    indep_pot = [key for key, value in conds.items() if (type(key) is v.StateVariable) and len(np.atleast_1d(value)) > 1]
    if (len(indep_comp) != 1) or (len(indep_pot) != 1):
        raise ValueError('Binary map requires exactly one composition and one potential coordinate')
    if indep_pot[0] != v.T:
        raise ValueError('Binary map requires that a temperature grid must be defined')

    # binary assumption, only one composition specified.
    comp_cond = [k for k in conds.keys() if isinstance(k, v.X)][0]
    indep_comp = comp_cond.name[2:]
    indep_comp_idx = sorted(comps).index(indep_comp)
    composition_grid = unpack_condition(conds[comp_cond])
    dX = composition_grid[1] - composition_grid[0]
    Xmax = composition_grid.max()
    temperature_grid = unpack_condition(conds[v.T])
    dT = temperature_grid[1] - temperature_grid[0]

    boundary_sets = boundary_sets or ZPFBoundarySets(comps, comp_cond)

    equilibria_calculated = 0
    equilibrium_time = 0
    convex_hulls_calculated = 0
    convex_hull_time = 0
    curr_conds = {key: unpack_condition(val) for key, val in conds.items()}
    for T in np.nditer(temperature_grid):
        iter_equilibria = 0
        if verbose:
            print("=== T = {} ===".format(float(T)))
        curr_conds[v.T] = [float(T)]
        eq_conds = deepcopy(curr_conds)
        Xmax_visited = 0.0
        hull_time = time.time()
        # TODO: try to refactor this to just use starting point generation, build my own grid
        hull = convex_hull(dbf, comps, phases, curr_conds, **eq_kwargs)
        grid = hull[-1]
        convex_hull_time += time.time() - hull_time
        convex_hulls_calculated += 1
        while Xmax_visited < Xmax:
            hull_compsets = find_two_phase_region_compsets(hull, T, indep_comp, indep_comp_idx, minimum_composition=Xmax_visited, misc_gap_tol=2*dX)
            if hull_compsets is None:
                if verbose:
                    print("== Convex hull: max visited = {} - no multiphase phase compsets found ==".format(Xmax_visited, hull_compsets))
                break
            Xeq = hull_compsets.mean_composition
            eq_conds[comp_cond] = [float(Xeq)]
            str_conds = sorted([str(k) for k in eq_conds.keys()])
            eq_time = time.time()
            start_point = starting_point(eq_conds, statevars, prxs, grid)
            eq_ds = _solve_eq_at_conditions(species, start_point, prxs, grid, str_conds, statevars, False)
            equilibrium_time += time.time() - eq_time
            equilibria_calculated += 1
            iter_equilibria += 1
            # composition sets in the plane of the calculation:
            # even for isopleths, this should always be two.
            compsets = get_compsets(eq_ds, indep_comp, indep_comp_idx)
            if verbose:
                print("== Convex hull: max visited = {:0.4f} - hull compsets: {} equilibrium compsets: {} ==".format(Xmax_visited, hull_compsets, compsets))
            if compsets is None:
                # equilibrium calculation, didn't find a valid multiphase composition set
                # we need to find the next feasible one from the convex hull.
                Xmax_visited += dX
                continue
            else:
                boundary_sets.add_compsets(compsets, Xtol=0.10, Ttol=2*dT)
                if compsets.max_composition > Xmax_visited:
                    Xmax_visited = compsets.max_composition
            # this seems kind of sloppy, but captures the effect that we want to
            # keep doing equilibrium calculations, if possible.
            while Xmax_visited < Xmax and compsets is not None:
                eq_conds[comp_cond] = [float(Xmax_visited + dX)]
                str_conds = sorted([str(k) for k in eq_conds.keys()])
                eq_time = time.time()
                # TODO: starting point could be improved by basing it off the previous calculation
                start_point = starting_point(eq_conds, statevars, prxs, grid)
                eq_ds = _solve_eq_at_conditions(species, start_point, prxs, grid, str_conds, statevars, False)
                equilibrium_time += time.time() - eq_time
                equilibria_calculated += 1
                compsets = get_compsets(eq_ds, indep_comp, indep_comp_idx)
                if compsets is not None:
                    Xmax_visited = compsets.max_composition
                    boundary_sets.add_compsets(compsets, Xtol=0.10, Ttol=2*dT)
                else:
                    Xmax_visited += dX
                if verbose:
                    print("Equilibrium: at X = {:0.4f}, found compsets {}".format(Xmax_visited, compsets))
        if verbose:
            print(iter_equilibria, 'equilibria calculated in this iteration.')
    if verbose or summary:
        print("{} Convex hulls calculated ({:0.2f}s)".format(convex_hulls_calculated, convex_hull_time))
        print("{} Equilbria calculated ({:0.0f}s)".format(equilibria_calculated, equilibrium_time))
        print("{:0.0f}% of brute force calculations skipped".format(100*(1-equilibria_calculated/(composition_grid.size*temperature_grid.size))))
    return boundary_sets
