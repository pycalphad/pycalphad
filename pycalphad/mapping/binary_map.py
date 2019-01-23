
from copy import deepcopy
from operator import pos, neg
import numpy as np
from pycalphad import equilibrium, variables as v
from .utils import get_num_phases, close_to_same, close_zero_or_one, get_compsets, opposite_direction
from .start_points import StartPoint, find_three_phase_start_points, find_nearby_region_start_point
from .zpf_boundary_sets import ZPFBoundarySets


def binplot_map(dbf, comps, phases, conds, tol_zero_one=None, tol_same=None, tol_misc_gap=0.1, eq_kwargs=None, max_T_backtracks=5, T_backtrack_factor=2, verbose=False, veryverbose=False, **plot_kwargs):
    # naive algorithm to map a binary phase diagram in T-X
    # for each temperature, proceed along increasing composition, skipping two phase regions
    # assumes conditions in T and X

    # assumes only one composition
    x_cond = [c for c in conds.keys() if isinstance(c, v.Composition)][0]
    # mapping conditions
    x_min, x_max, dx = conds[x_cond]
    T_min, T_max, dT = conds[v.T]
    curr_conds = deepcopy(conds)

    start_points = []

    # find a starting point
    # TODO: use convex hull to find the first two phase region from the left
    starting_x = 0.1*(x_max - x_min)+x_min
    starting_T = starting_T_max = 0.9*(T_max - T_min)+T_min
    while len(start_points) == 0:
        curr_conds[v.T] = starting_T
        curr_conds[x_cond] = starting_x
        eq_res = equilibrium(dbf, comps, phases, curr_conds)
        if get_num_phases(eq_res) == 2:
            # add a direction of dT > 0 and dT < 0
            start_points.append(StartPoint(starting_T, pos, get_compsets(eq_res)))
            start_points.append(StartPoint(starting_T, neg, get_compsets(eq_res)))

        if starting_T - dT > T_min:
            starting_T -= dT
        else:
            starting_T = starting_T_max
            starting_x += dx

    zpf_boundaries = ZPFBoundarySets()
    tol_zero_one = tol_zero_one if tol_zero_one is not None else dx  # convergence tolerance
    tol_same = tol_same if tol_same is not None else dx
    # Main loop
    while len(start_points) > 0:
        start_pt = start_points.pop()
        delta = start_pt.direction(dT)

        prev_compsets = start_pt.compsets
        d_str = "+" if start_pt.direction is pos else "-"
        if verbose:
            print("Entering region {} in the {} direction".format(prev_compsets, d_str))
        T_current = start_pt.temperature + delta
        x_current = start_pt.composition
        T_backtracks = 0; total_T_backtrack_factor = 1;
        converged = False
        while not converged:
            if (T_current < T_min) or (T_current > T_max):
                converged = True
                continue
            curr_conds[v.T] = T_current
            curr_conds[x_cond] = x_current
            eq = equilibrium(dbf, comps, phases, curr_conds)
            compsets = get_compsets(eq)
            if veryverbose:
                print("found compsets {} at T={}K X={} eq_phases={}".format(compsets, T_current, x_current, eq.Phase.values.flatten()))
            if len(compsets) == 1:
                found_str = "Found single phase region {} at T={}K X={}".format(compsets[0].phase_name, T_current, x_current)
                if T_backtracks < max_T_backtracks:
                    T_backtracks += 1
                    total_T_backtrack_factor *= T_backtrack_factor
                    T_backtrack = T_current - delta/total_T_backtrack_factor
                    if verbose:
                        print(found_str + " Backtracking in temperature from {}K to {}K ({}/{})".format(T_current, T_backtrack, T_backtracks, max_T_backtracks))
                    T_current = T_backtrack
                    continue
                else:
                    raise ValueError("Mapping error:" + found_str)
            elif len(compsets) >= 3:
                raise ValueError("Mapping error: found {} phases ({}) instead of 2".format(len(compsets), "/".join([c.phase_name for c in compsets])))
            else:
                T_backtracks = 0; total_T_backtrack_factor = 1
            zpf_boundaries.add_compsets(*compsets)
            cs_0 = compsets[0].composition
            cs_1 = compsets[1].composition
            if close_zero_or_one(cs_0, tol_zero_one) and close_zero_or_one(cs_1, tol_zero_one):
                converged = True
                continue
            if close_to_same(cs_0, cs_1, tol_same):
                converged = True
                # find other two phase equilibrium
                new_start_point = find_nearby_region_start_point(dbf, comps ,phases, compsets, zpf_boundaries, T_current, dT, deepcopy(curr_conds), x_cond, verbose=verbose)
                if verbose:
                    print("New start point {} from convergence to same value at T={}K and X={}".format(new_start_point, T_current, x_current))
                if new_start_point is not None:
                    zpf_boundaries.add_compsets(*new_start_point.compsets)
                    start_points.append(new_start_point)
                continue

            prev_phases = {c.phase_name for c in prev_compsets}
            curr_phases = {c.phase_name for c in compsets}
            common_phases = curr_phases.intersection(prev_phases)
            new_phases = curr_phases - common_phases
            if len(new_phases) == 1: # we found a new phase!
                new_start_points = find_three_phase_start_points(compsets, prev_compsets, start_pt.direction)
                if verbose:
                    print("New start points {} from three phase equil. at T={}K and X={}".format(new_start_points, T_current, x_current))
                start_points.extend(new_start_points)
                converged = True
                continue
            elif len(new_phases) > 1:
                raise ValueError("Found more than 1 new phase")
            elif len(new_phases) == 0:  # TODO: this could get expensive, we hit it every time
                # we have the same phases
                # check that the composition of any two phases didn't change significantly
                # if there is significant change, there may be a miscibility gap.
                # add a start point at the current temperature in the opposite direction
                for cs in prev_compsets:
                    matching_compsets = [c for c in compsets if c.phase_name == cs.phase_name]
                    if len(matching_compsets) == 1:
                        # we are not currently in a miscibility gap
                        matching_cs = matching_compsets[0]
                        same_phase_comp_diff = np.abs(cs.composition - matching_cs.composition)
                        if same_phase_comp_diff > tol_misc_gap:
                            if verbose:
                                print("Found potential miscibility gap compsets {} differ in composition by {}".format([cs, matching_cs], same_phase_comp_diff))
                            start_points.append(StartPoint(T_current, opposite_direction(start_pt.direction), [cs, matching_cs]))

            T_current += delta
            x_current = np.mean([c.composition for c in compsets])
            prev_compsets = compsets
    return zpf_boundaries
