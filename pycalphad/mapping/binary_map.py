
import time
from copy import deepcopy
import numpy as np
from pycalphad import equilibrium, variables as v
from pycalphad.core.utils import unpack_condition
from pycalphad.codegen.callables import build_callables
from .compsets import CompSet2D
from .utils import get_compsets, convex_hull, find_two_phase_region_compsets
from .zpf_boundary_sets import ZPFBoundarySets

class StartingPointError(Exception):
    pass

def binplot_map(dbf, comps, phases, conds, eq_kwargs=None, verbose=False, boundary_sets=None):
    """

    Parameters
    ----------
    dbf :
    comps :
    phases :
    conds :
    eq_kwargs :
    verbosity :
    boundary_sets :

    Returns
    -------
    ZPFBoundarySets

    Notes
    -----
    Naïve algorithm to map a binary phase diagram in T-X
    for each temperature, proceed along increasing composition, skipping two phase regions
    assumes conditions in T and X
    Right now, this is assumed to be a binary system, but it's feasible to accept
    a set of constraints in the conditions for compositions that specify an
    ispleth in multicomponent space, and this code will transform so that X
    follows the path with the constraints, transforming the equilibrium hyperplanes
    as necessary.

    """

    # binary assumption, only one composition specified.
    comp_cond = [k for k in conds.keys() if isinstance(k, v.X)][0]
    # TODO: In the general case, we need this and the index to be replaced with
    #       a function to calculate the mapping composition based on all the
    #       pure element compositions and the constraints.
    indep_comp = comp_cond.name[2:]
    indep_comp_idx = sorted(comps).index(indep_comp)
    composition_grid = unpack_condition(conds[comp_cond])
    dX = composition_grid[1] - composition_grid[0]
    Xmax = composition_grid.max()
    temperature_grid = unpack_condition(conds[v.T])

    boundary_sets = boundary_sets or ZPFBoundarySets(comps, comp_cond)

    curr_conds = deepcopy(conds)
    for T in np.nditer(temperature_grid):
        if verbose:
            print("=== T = {} ===".format(float(T)))
        curr_conds[v.T] = float(T)
        eq_conds = deepcopy(curr_conds)
        Xmax_visited = 0.0
        hull = convex_hull(dbf, comps, phases, curr_conds, **eq_kwargs)
        while Xmax_visited < Xmax:
            hull_compsets = find_two_phase_region_compsets(hull, T, indep_comp, indep_comp_idx, minimum_composition=Xmax_visited)
            if len(hull_compsets) == 0:
                if verbose:
                    print("== Convex hull: max visited = {} - no multiphase phase compsets found ==".format(Xmax_visited, hull_compsets))
                break
            Xeq = CompSet2D.mean_composition(hull_compsets)
            eq_conds[comp_cond] = Xeq
            eq_ds = equilibrium(dbf, comps, phases, eq_conds, **eq_kwargs)
            # composition sets in the plane of the calculation:
            # even for isopleths, this should always be two.
            compsets = get_compsets(eq_ds, indep_comp, indep_comp_idx)
            if verbose:
                print("== Convex hull: max visited = {} - hull compsets: {} equilibrium compsets: {} ==".format(Xmax_visited, hull_compsets, compsets))
            if len(compsets) < 2:
                # equilibrium calculation, didn't find a valid multiphase composition set
                # we need to find the next feasible one from the convex hull.
                Xmax_visited += dX
                continue
            # this seems kind of sloppy, but captures the effect that we want to
            # keep doing equilibrium calculations, if possible.
            while Xmax_visited < Xmax and len(compsets) == 2:
                # TODO: This might not be necessary, but we're playing it safe
                #       for now. This is the result of an old design where we
                #       did only specific two phase regions at a time.
                boundary_sets.add_boundary_set()
                boundary_sets.add_compsets(compsets)
                Xmax_visited = CompSet2D.max_composition(compsets) + dX
                eq_conds[comp_cond] = Xmax_visited
                eq_ds = equilibrium(dbf, comps, phases, eq_conds, **eq_kwargs)
                compsets = get_compsets(eq_ds, indep_comp, indep_comp_idx)
                if verbose:
                    print("Equilibrium: at X = {}, found compsets {}".format(Xmax_visited, compsets))

    return boundary_sets
