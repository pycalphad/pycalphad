
import time
from copy import deepcopy
import numpy as np
from pycalphad import equilibrium, variables as v
from pycalphad.codegen.callables import build_callables
from .compsets import BinaryCompSet
from .utils import close_to_same, close_zero_or_one, get_compsets, opposite_direction, convex_hull, find_two_phase_region_compsets, Direction
from .start_points import StartPoint, find_three_phase_start_points, find_nearby_region_start_point, StartPointsList
from .zpf_boundary_sets import ZPFBoundarySets

class StartingPointError(Exception):
    pass

def binplot_map(dbf, comps, phases, conds, tol_zero_one=None, tol_same=None, tol_misc_gap=0.1, eq_kwargs=None,
                max_T_backtracks=5, T_backtrack_factor=2, backtrack_raise=False,
                startpoint_comp_tol=0.05, startpoint_temp_tol=20, verbosity=0,
                initial_start_points=None):
    # naive algorithm to map a binary phase diagram in T-X
    # for each temperature, proceed along increasing composition, skipping two phase regions
    # assumes conditions in T and X

    eq_kwargs = eq_kwargs or {}
    if 'callables' not in eq_kwargs:
        eq_kwargs['callables'] = build_callables(dbf, comps, phases, build_gradients=True, model=eq_kwargs.get('model'))
        eq_kwargs['model'] = eq_kwargs['callables']['model']
    # assumes only one composition
    x_cond = [c for c in conds.keys() if isinstance(c, v.Composition)][0]
    indep_comp = x_cond.species.name
    comp_idx = sorted(set(comps) - {'VA'}).index(indep_comp)
    # mapping conditions
    x_min, x_max, dx = conds[x_cond]
    T_min, T_max, dT = conds[v.T]
    curr_conds = deepcopy(conds)
    tol_zero_one = tol_zero_one if tol_zero_one is not None else dx  # convergence tolerance
    tol_same = tol_same if tol_same is not None else dx
    zpf_boundaries = ZPFBoundarySets(comps, x_cond)

    start_points = StartPointsList(eq_comp_tol=startpoint_comp_tol, eq_temp_tol=startpoint_temp_tol)
    if initial_start_points is not None:
        if isinstance(initial_start_points, StartPoint):
            start_points.add_start_point(initial_start_points)
        else:
            # assume an iterable
            for sp in initial_start_points:
                start_points.add_start_point(sp)

    # find a starting point
    starting_T = 0.9*(T_max - T_min)+T_min
    time_start = time.time()
    max_startpoint_discrepancy = np.max([tol_zero_one, tol_same, dx])
    while len(start_points.remaining_start_points) < 1:
        curr_conds[v.T] = starting_T
        hull = convex_hull(dbf, comps, phases, curr_conds, **eq_kwargs)
        cs = find_two_phase_region_compsets(hull, starting_T, indep_comp, comp_idx, discrepancy_tol=max_startpoint_discrepancy)
        if len(cs) == 2:
            # verify that these show up in the equilibrium calculation
            specific_conds = deepcopy(curr_conds)
            specific_conds[x_cond] = BinaryCompSet.mean_composition(cs)
            eq_cs = get_compsets(equilibrium(dbf, comps, phases, specific_conds, **eq_kwargs), indep_comp=indep_comp, indep_comp_index=comp_idx)
            if len(eq_cs) == 2:
                # add a direction of dT > 0 and dT < 0
                zpf_boundaries.add_compsets(eq_cs)
                # shift starting_T so they start at the same place.
                start_points.add_start_point(StartPoint(starting_T - dT, Direction.POSITIVE, eq_cs))
                start_points.add_start_point(StartPoint(starting_T + dT, Direction.NEGATIVE, eq_cs))

        if starting_T - dT > T_min:
            starting_T -= dT
        else:
            raise StartingPointError("Unable to find an initial starting point.")
    if verbosity >= 1:
        print("Found start points {} in {:0.2f}s".format(start_points, time.time()-time_start))

    # Main loop
    while len(start_points.remaining_start_points) > 0:
        zpf_boundaries.add_boundary_set()
        start_pt = start_points.get_next_start_point()
        curr_direction = start_pt.direction
        delta = curr_direction*dT

        prev_compsets = start_pt.compsets
        if verbosity >= 1:
            print("Entering region {}".format(start_pt))
        T_current = start_pt.temperature + delta
        x_current = start_pt.composition
        T_backtracks = 0; total_T_backtrack_factor = 1;
        converged = False
        while not converged:
            if (T_current < T_min) or (T_current > T_max):
                converged = True
                end_point = StartPoint(T_current, opposite_direction(curr_direction), compsets)
                start_points.add_end_point(end_point)
                if verbosity >= 2:
                    print("Terminating at end point from temperature outside of bounds ({}, {}) at {}".format(T_min, T_max, end_point))
                continue
            curr_conds[v.T] = T_current
            curr_conds[x_cond] = x_current
            eq = equilibrium(dbf, comps, phases, curr_conds, **eq_kwargs)
            compsets = get_compsets(eq, indep_comp=indep_comp, indep_comp_index=comp_idx)
            if verbosity >= 3:
                print("found compsets {} at T={}K X={:0.3f} eq_phases={}".format(compsets, T_current, x_current, eq.Phase.values.flatten()))
            if len(compsets) == 1:
                found_str = "Found single phase region {} at T={}K X={:0.3f}".format(compsets[0].phase_name, T_current, x_current)
                if T_backtracks < max_T_backtracks:
                    T_backtracks += 1
                    total_T_backtrack_factor *= T_backtrack_factor
                    T_backtrack = T_current - delta/total_T_backtrack_factor
                    if verbosity >= 2:
                        print(found_str + " Backtracking in temperature from {}K to {}K ({}/{})".format(T_current, T_backtrack, T_backtracks, max_T_backtracks))
                    T_current = T_backtrack
                    continue
                elif not backtrack_raise:
                    converged = True
                    end_point = StartPoint(T_current, opposite_direction(curr_direction), compsets)
                    start_points.add_end_point(end_point)
                    if verbosity >= 2:
                        print("Terminating at end point due to single phase equilibrium: {}".format(end_point))
                    # We might be stuck near a congruent point.
                    # Try to do a nearby search using the last known compsets
                    # If we can't find one, just continue.
                    T_prev = prev_compsets[0].temperature
                    new_start_point = find_nearby_region_start_point(dbf, comps, phases, prev_compsets, comp_idx, T_prev, dT, deepcopy(curr_conds), x_cond, start_points, verbose=True if verbosity >= 1 else False, hull_kwargs=eq_kwargs)
                    if new_start_point is not None:
                        if verbosity >= 1:
                            print("Failed to backtrack. Found new start point {} from convergence-like to same value at T={}K and X={:0.3f}".format(new_start_point, T_prev, x_current))
                        zpf_boundaries.add_compsets(new_start_point.compsets)
                        start_points.add_start_point(new_start_point)
                    continue
                else:
                    raise ValueError("Mapping error:" + found_str + " Last two phase region: {}".format(prev_compsets))
            elif len(compsets) >= 3:
                raise ValueError("Mapping error: found {} phases ({}) instead of 2".format(len(compsets), "/".join([c.phase_name for c in compsets])))
            else:
                T_backtracks = 0; total_T_backtrack_factor = 1
            cs_0 = compsets[0].composition
            cs_1 = compsets[1].composition
            if close_zero_or_one(cs_0, tol_zero_one) and close_zero_or_one(cs_1, tol_zero_one):
                converged = True
                zpf_boundaries.add_compsets(compsets)
                if verbosity >= 2:
                    print("Terminating at composition near 0 or 1 with composition sets {}".format(compsets))
                continue
            if close_to_same(cs_0, cs_1, tol_same):
                converged = True
                end_point = StartPoint(T_current, opposite_direction(curr_direction), compsets)
                start_points.add_end_point(end_point)
                zpf_boundaries.add_compsets(compsets)
                # find other two phase equilibrium
                new_start_point = find_nearby_region_start_point(dbf, comps ,phases, compsets, comp_idx, T_current, dT, deepcopy(curr_conds), x_cond, start_points, verbose=True if verbosity >= 1 else False, hull_kwargs=eq_kwargs)
                if new_start_point is not None:
                    if verbosity >= 1:
                        print("New start point {} from convergence to same value at T={}K and X={}".format(new_start_point, T_current, x_current))
                if verbosity >= 2:
                    print("Terminating at end point because of convergence to same value {}".format(end_point))
                continue

            prev_phases = {c.phase_name for c in prev_compsets}
            curr_phases = {c.phase_name for c in compsets}
            common_phases = curr_phases.intersection(prev_phases)
            new_phases = curr_phases - common_phases
            if len(new_phases) == 1: # we found a new phase!
                converged = True
                end_point = StartPoint(T_current - delta, opposite_direction(curr_direction), prev_compsets)
                start_points.add_end_point(end_point)
                if verbosity >= 2:
                    print("Terminating at three phase equilibria with end point {}".format(end_point))
                new_start_points = find_three_phase_start_points(compsets, prev_compsets, curr_direction)
                if verbosity >= 1:
                    print("New start points {} from three phase equilibrium at T={}K and X={}".format(new_start_points, T_current, x_current))
                for sp in new_start_points:
                    start_points.add_start_point(sp)
                # don't need to add new compsets here because they will be picked up based on the new start points
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
                        same_phase_comp_diff = cs.xdiscrepancy(matching_cs)
                        if same_phase_comp_diff > tol_misc_gap:
                            converged = True
                            sp_gap = StartPoint(T_current, opposite_direction(curr_direction), [cs, matching_cs])
                            start_points.add_start_point(sp_gap)
                            # we need to start a new start point, otherwise there will be a boundary line that "jumps" the miscibility gap
                            # the tradeoff is that there is a break in the boundaries
                            sp_continue = StartPoint(T_current-delta, curr_direction, compsets)
                            start_points.add_start_point(sp_continue)
                            if verbosity >= 1:
                                print("New start points {} from miscibility gap".format([sp_gap, sp_continue]))
                            if verbosity >= 2:
                                print("Terminating at miscibility gap.")
            if converged:
                continue
            zpf_boundaries.add_compsets(compsets)
            T_current += delta
            x_current = BinaryCompSet.mean_composition(compsets)
            prev_compsets = compsets
    return zpf_boundaries
