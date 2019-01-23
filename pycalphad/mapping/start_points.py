import numpy as np
from operator import pos, neg
from .utils import convex_hull, get_compsets, sort_x_by_y
from pycalphad import variables as v
import xarray as xr

class StartPoint():
    def __init__(self, temperature, direction, compsets, composition=None):
        self.temperature = temperature
        self.direction = direction
        self.compsets = compsets
        if composition is not None:
            self.composition = composition
        else:
            # get the average composition from the compsets
            self.composition = np.mean([c.composition for c in compsets])

    def __repr__(self):
        phases = "/".join([c.phase_name for c in self.compsets])
        if self.direction is pos:
            dir_str = "+"
        else:
            dir_str = "-"
        return "StartPoint<T={}, dT=({}), X={}, Phases={}>".format(
            self.temperature, dir_str, self.composition, phases)


def find_three_phase_start_points(new_compsets, prev_compsets, direction):
    """
    Returns two new start points from a three phase invariant reaction

    Notes
    -----
    This is around a three phase equilibrium invariant reaction.

    There are two situations:
    1. Two two-phase regions above, one two-phase region below (eutectic)
    2. One two-phase region above, two two-phase regions below (peritectic)

    Graphically, an invariant reaction occurs at `=`

    Situation 1 (eutectic-like)
    |------------------|
    |------------------|
    |------------------|
    |------------------|
    |========|=========|
    |--------|---------|
    |--------|---------|
    |--------|---------|
    |--------|---------|

    Situation 2 (peritectic-like)
    |--------|---------|
    |--------|---------|
    |--------|---------|
    |--------|---------|
    |--------|---------|
    |========|=========|
    |------------------|
    |------------------|
    |------------------|
    |------------------|

    We need to find all three regions, remove the one we have mapped already
    and add the two new ones. The best way to do this would be to
    calculate three phase equilibrium, since we cannot do this
    in pycalphad currently, we have to approximate it.

    We exploit the fact that we are always transferring between a large and small region.
    Thus we add the start point for the new region we found (in the same direction) and we
    always have to find the opposing small region. If we go from a large to small region,
    the opposing small region should be in the same direction, otherwise if we go from a small
    to large region, the new small region direction should have the opposite direction.
    """
    prev_phases = [c.phase_name for c in prev_compsets]
    prev_comps = [c.composition for c in prev_compsets]
    prev_comps_diff = np.abs(np.max(prev_comps) - np.min(prev_comps))

    new_phases = [c.phase_name for c in new_compsets]
    new_comps = [c.composition for c in new_compsets]
    new_comps_diff = np.abs(np.max(new_comps) - np.min(new_comps))

    # In all cases, we want a new StartPoint for the new compsets in the direction we were going
    start_points = [StartPoint(new_compsets[0].temperature, direction, new_compsets)]

    # assign small and large regions
    if (new_comps_diff < prev_comps_diff):  # went from large to small region
        L_cs = prev_compsets  # large region
        S_cs = new_compsets  # small region
        L_phases = prev_phases
        S_phases = new_phases
        new_direction = pos if direction is pos else neg
    else:  # went from small to large region
        L_cs = new_compsets  # large region
        S_cs = prev_compsets  # small region
        L_phases = new_phases
        S_phases = prev_phases
        new_direction = pos if direction is neg else neg

    opposing_small_region_cs = [c for c in S_cs if c.phase_name not in L_phases] + [c for c in L_cs if c.phase_name not in S_phases]
    start_points.append(StartPoint(S_cs[0].temperature, new_direction, opposing_small_region_cs))

    return start_points


def find_nearby_region_start_point(dbf, comps ,phases, compsets, zpf_boundaries, temperature, dT,
                                   conds, indep_comp_cond, cutoff_search_distance=0.1,
                                   verbose=False, graceful=True):
    """
    Return a starting point for a nearby region.

    Parameters
    ----------
    compsets : list
    cutoff_search_distance : float
        Distance in composition to cutoff the search for new phases.

    The idea here is that the compsets have converged to each other (e.g. at a congruent melting point)
    and we've mapped out one side of the point and need to find the other side.

    The idea is that we select several temperatures and construct a convex hull in composition
    at those temperature to search the composition region. Then we will go through the points in overall composition
    from nearest to farthest from the average composition and try to find where there is
    1. Two phases in equilibrium
    2. At least one common phase with the current equilibrium
    3. The ordering of the phases w.r.t composition are different e.g. (X(LIQUID)>X(CU2MG) in one set vs. X(LIQUID)<X(CU2MG) in another set)
    for the positive and negative directions.
    """
    current_phases = [c.phase_name for c in compsets]
    current_phases_set = set(current_phases)
    compositions = [c.composition for c in compsets]
    str_comp = str(indep_comp_cond)
    average_comp = np.mean(compositions)
    sorted_phases = sort_x_by_y(current_phases, compositions)  # phases sorted by min to max composition

    # first we'll search temperatures very close to the current temperature (shifted by dT/10, then we'll do a full dT, then dT+dT/10)
    trial_Ts = [
        (temperature - dT / 10.0, neg),
        (temperature + dT / 10.0, pos),
        (temperature - dT, neg),
        (temperature + dT, pos),
        (temperature - dT - dT / 10.0, neg),
        (temperature + dT + dT / 10.0, pos),
    ]

    # take the first result we get
    for trial_T, trial_direction in trial_Ts:
        conds[v.T] = trial_T
        conds[indep_comp_cond] = (0, 1, 0.005)  # composition grid
        hull = convex_hull(dbf, comps, phases, conds)
        hull = hull.sortby(np.abs(hull[str_comp] - average_comp))
        # TODO: use masking on the composition cutoff so that find_two_phase_region_compsets can be used
        for i in range(hull.sizes[str_comp]):
            if np.abs(hull[str_comp][i] - average_comp) > cutoff_search_distance:
                break
            cur_hull = hull.isel({str_comp: i})
            trial_compsets = get_compsets(cur_hull)
            trial_phases = [c.phase_name for c in trial_compsets]
            trial_phases_set = set(trial_phases)
            trial_compositions = [c.composition for c in trial_compsets]
            sorted_trial_phases = sort_x_by_y(trial_phases, trial_compositions)
            # Convex hull always gives back pairs of compsets, even for true single phase regions.
            # We need to filter out regions where the phases aren't the same, those aren't true two phase regions.
            # This might break in a miscibility gap.
            # Condition 1: Number of phases must be 2
            if len(trial_phases_set) != 2:
                continue
            # Condition 2: Must share one unique phase
            if len(current_phases_set.intersection(trial_phases_set)) < 1:
                continue
            # Condition 3: Ordering of the set of phases must be different
            if sorted_phases == sorted_trial_phases:
                continue
            # If we made it here, we found a match!
            sp = StartPoint(trial_T, trial_direction, trial_compsets)
            # Don't add boundaries because this is an inaccurate set
            # zpf_boundaries.add_compsets(*trial_compsets)
            return sp
    if graceful:
        return
    else:
        raise ValueError( "Could not find start point for neighbor to compsets: {}".format(compsets))
