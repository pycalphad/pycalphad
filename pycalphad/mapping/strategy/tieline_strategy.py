from typing import Union
import logging
import itertools
import copy

import numpy as np

from pycalphad import Database, variables as v
from pycalphad.core.composition_set import CompositionSet
from pycalphad.property_framework.computed_property import LinearCombination
from pycalphad.property_framework.units import to_display_units

from pycalphad.mapping.primitives import ZPFLine, Node, Point, ExitHint, Direction, MIN_COMPOSITION, ZPFState, _eq_compset, _get_phase_specific_variable
from pycalphad.mapping.starting_points import point_from_equilibrium
import pycalphad.mapping.zpf_equilibrium as zeq
import pycalphad.mapping.utils as map_utils
from pycalphad.mapping.strategy.strategy_base import MapStrategy
from pycalphad.mapping.strategy.step_strategy import StepStrategy
from pycalphad.mapping.strategy.strategy_data import SinglePhaseData, StrategyData, PhaseRegionData

_log = logging.getLogger(__name__)

def _sort_point_potential_axis(point: Point, axis_vars: list[v.StateVariable], norm: dict[v.StateVariable, float]):
    """
    Given a point with 2 free phases when mapping with a potential axis, get derivative at both
    composition sets to test which CS to fix and which direction to start
    """
    _log.info(f"Sorting point {point.fixed_phases}, {point.free_phases}, {point.global_conditions}")

    # Sort axis by state variables first
    # This shouldn't be necessary since dT/dX return np.nan, which we replace to np.inf for getting the optimal direction
    # But just in case, we default to stepping in temperature if no direction could be found
    axis_vars = map_utils._sort_axis_by_state_vars(axis_vars)

    # Remove first axis variable from point - assumes we default to stepping along second variable
    # We keep a copy of the conditions in case we decide that stepping along the first variable is better
    comp_sets = point.stable_composition_sets

    # 4 options here as a combination of
    #    Fix CS 1 and free CS 2, or fix CS 2 and free CS 1
    #    d(av1)/d(av2) or d(av2)/d(av1)
    cs_options = [[comp_sets[0], comp_sets[1]], [comp_sets[1], comp_sets[0]]]
    av_options = [[axis_vars[0], axis_vars[1]], [axis_vars[1], axis_vars[0]]]
    options_tests = []
    for options in itertools.product(cs_options, av_options):
        cs_list, av_list = options
        new_point = map_utils._generate_point_with_fixed_cs(point, cs_list[0], cs_list[1])
        der = abs(zeq.compute_derivative(new_point, av_list[0], av_list[1]))
        if np.isnan(der):
            der = np.inf
        else:
            # Normalize derivative (normalization factor should be is axis delta)
            der *= norm[av_list[1]] / norm[av_list[0]]
        options_tests.append((der, new_point, av_list[0]))

    # Best point/axis var is determined by the lowest derivative over 1
    best_index = -1
    best_der = np.inf
    for i in range(len(options_tests)):
        if options_tests[i][0] > 1:
            _log.info(f"Option: Axis var {options_tests[i][2]}, derivative {options_tests[i][0]}, point {options_tests[i][1].fixed_phases}, {options_tests[i][1].free_phases}")
            if options_tests[i][0] <= best_der:
                best_index = i
                best_der = options_tests[i][0]

    # If no best point/axis var, then use the first one (which will be along a state variable)
    if best_index == -1:
        return options_tests[0][1], options_tests[0][2]
    else:
        return options_tests[best_index][1], options_tests[best_index][2]

def _get_delta_cs_var(point: Point, comp_sets: list[CompositionSet], axis_vars: list[v.StateVariable], normalize=True):
    """
    Given two composition sets, get the vector using axis_vars as coordinates

    Seems a bit strange to have the comp sets as an input when we already have the point, but it's just to have some
    flexibility on which comp sets to treat at 0 or 1
    """
    dx = point.get_local_property(comp_sets[0], axis_vars[0]) - point.get_local_property(comp_sets[1], axis_vars[0])
    dy = point.get_local_property(comp_sets[0], axis_vars[1]) - point.get_local_property(comp_sets[1], axis_vars[1])
    norm_factor = 1 if not normalize else np.sqrt(dx*dx + dy*dy)
    return [dx/norm_factor, dy/norm_factor]

def _get_norm(point: Point, axis_vars: list[v.StateVariable]):
    """
    Creates normal vector of a tieline

    Assumes point has two phases
    """
    vec = _get_delta_cs_var(point, point.stable_composition_sets, axis_vars)
    return [-vec[1], vec[0]]

def _create_linear_comb_conditions(point: Point, axis_vars: list[v.StateVariable], normal: list[float] = None):
    """
    Creates a linear combination along a normal vector

    If normal is not given, then the normal will be the normal of the tieline defined by the point
    """
    # Get normal and axis variable to step in (this will be along the maximum of the normal vector)
    if normal is None:
        normal = _get_norm(point, axis_vars)

    # Create a linear combination that forces the stepping to be along the normal
    try:
        av1_value = point.get_property(axis_vars[1])
    except ValueError:
        # A node can consist solely of fixed composition sets with zero phase amount, where
        # the global composition cannot be recovered from the phases (0/0). The node's
        # coordinates are bookkept in its conditions, so anchor the line there instead.
        if axis_vars[1] in point.global_conditions:
            av1_value = point.global_conditions[axis_vars[1]]
        else:
            raise
    c = normal[1]*point.global_conditions[axis_vars[0]] - normal[0]*av1_value
    lc = LinearCombination(normal[1]*axis_vars[0] - normal[0]*axis_vars[1])
    return lc, c

def _sort_point_composition_axes(point: Point, axis_vars: list[v.StateVariable]):
    """
    Given a point with 2 free phases when mapping with only composition axes, get derivative
    at both composition sets to test which CS to fix and which direction to start

    NOTE: unlike the potential axis version of this function, normal is defined from the tieline
          rather than the axis variables
    """
    _log.info(f"Sorting point {point.fixed_phases}, {point.free_phases}, {point.global_conditions}")

    # Free all phases in point, we will fix the phase at the end
    free_point = map_utils._generate_point_with_free_cs(point)

    # Get normal and axis variable to step in (this will be along the maximum of the normal vector)
    normal = _get_norm(free_point, axis_vars)
    av = axis_vars[np.argmax(np.abs(normal))]
    _log.info(f"Point {point.fixed_phases}, {point.free_phases} with normal {normal}. Testing derivative in {av}")

    # This is here to have derivative with respect to the axis variable to follow the direction of the normal
    #  NOTE: the derivative we compute is not the directional derivative with respect to the normal
    #  but rather the derivative with respect to the axis variable, with the normal as a fixed condition
    #  Since we lose the direction information of the normal when taking the derivative, mult_factor is here
    #  to retain that information
    mult_factor = np.sign(normal[axis_vars.index(av)])

    # Create a linear combination that forces the stepping to be along the normal
    lc, c = _create_linear_comb_conditions(point, axis_vars, normal)

    # Create temporary set of conditions that replaces the fixed axis variable with the linear combination
    del free_point.global_conditions[axis_vars[1-axis_vars.index(av)]]
    free_point.global_conditions[lc] = c

    # Test phase composition derivative for each phase stepping along the axis variable fixed along the normal
    stable_cs = free_point.stable_composition_sets
    cs_options = [[stable_cs[0], stable_cs[1]], [stable_cs[1], stable_cs[0]]]
    options_tests = []
    for cs_list in cs_options:
        for av_test in axis_vars:
            v_num = v.X(cs_list[1].phase_record.phase_name, av_test.species)
            der = mult_factor*zeq.compute_derivative(free_point, v_num, av, free_den = False)
            new_point = map_utils._generate_point_with_fixed_cs(point, cs_list[0], cs_list[1])
            options_tests.append((der, new_point, av_test))

    # Best index is the one with the largest derivative
    best_index = -1
    best_der = -1
    for i in range(len(options_tests)):
        _log.info(f"Option: Axis var {options_tests[i][2]}, derivative {options_tests[i][0]}, point {options_tests[i][1].fixed_phases}, {options_tests[i][1].free_phases}")
        if abs(options_tests[i][0]) > best_der:
            best_index = i
            best_der = abs(options_tests[i][0])

    if best_index == -1:
        return options_tests[0][0], options_tests[0][1], options_tests[0][2], normal
    else:
        return options_tests[best_index][0], options_tests[best_index][1], options_tests[best_index][2], normal

class TielineStrategy(MapStrategy):
    """
    Map strategy for phase diagrams where tie-lines lie in the plane of the two axis variables

    This covers:

    - Binary phase diagrams (1 composition axis, 1 potential axis)
    - Ternary isothermal sections (2 composition axes)
    - Pseudo-binary sections (1 composition axis, 1 potential axis, with the remaining
      degrees of freedom fixed by potential conditions, e.g. chemical potentials).
      The caller is responsible for ensuring that tie-lines lie in the mapping plane

    Where mapping with and without a potential axis behave differently, methods dispatch
    on ``self.num_potential_condition``.

    NOTE: this strategy assumes two axis variables
    """
    def __init__(self, dbf: Database, components: list[str], phases: list[str], conditions: dict[v.StateVariable, Union[float, tuple[float]]], **kwargs):
        super().__init__(dbf, components, phases, conditions, **kwargs)
        if self.num_potential_condition == 0:
            # The remaining (dependent) composition variable of the mapping simplex
            axis_species = {str(av.species) for av in self.axis_vars}
            mu_species = {str(key.species) for key in self.conditions if isinstance(key, v.ChemicalPotential)}
            unlisted_components = [c for c in self.components if str(c) not in {'VA'} | axis_species | mu_species]
            self.all_vars = self.axis_vars + [v.X(unlisted_components[0])]

    def generate_automatic_starting_points(self):
        """
        Searches axis limits to find starting points

        Here, we do a step mapping along the axis bounds and grab all the nodes
        The nodes of a step map is distinguished from starting points in that they have a parent
        """
        map_kwargs = self._constant_kwargs()

        if self.num_potential_condition > 0:
            # Iterate through axis variables, and set conditions to fix axis variable at min or max
            for av in self.axis_vars:
                for av_val in self.axis_lims[av]:
                    conds = copy.deepcopy(self.conditions)
                    conds[av] = av_val

                    # Coarse search (will need to make sure stepping works for very coarse searches as it can miss some nodes)
                    other_av = self._other_av(av)
                    av_range = np.amax(self.axis_lims[other_av]) - np.amin(self.axis_lims[other_av])
                    conds[other_av] = (self.axis_lims[other_av][0], self.axis_lims[other_av][1], av_range/20)

                    # Adjust composition conditions to be slightly above 0 or below 1 for numerical stability
                    if isinstance(av, v.X):
                        if conds[av] == 0:
                            conds[av] = MIN_COMPOSITION
                        elif conds[av] == 1:
                            conds[av] = 1 - MIN_COMPOSITION

                    # Step map
                    step = StepStrategy(self.dbf, self.components, self.phases, conds, **map_kwargs)
                    step.do_map()
                    self.add_starting_points_from_step(step)
        else:
            # Iterate through axis variables, and set conditions to fix axis variable at min only
            for av in self.axis_vars:
                conds = copy.deepcopy(self.conditions)
                conds[av] = np.amin(self.axis_lims[av])

                # Adjust composition conditions to be slightly above 0 for numerical stability
                if isinstance(av, v.X):
                    if conds[av] == 0:
                        conds[av] = MIN_COMPOSITION

                # Step map
                step = StepStrategy(self.dbf, self.components, self.phases, conds, **map_kwargs)
                step.do_map()
                self.add_starting_points_from_step(step)

            # Additional step where we switch axis conditions to the unlisted variable
            conds = copy.deepcopy(self.conditions)
            conds[self.all_vars[-1]] = MIN_COMPOSITION
            del conds[self.axis_vars[0]]

            # Step map
            step = StepStrategy(self.dbf, self.components, self.phases, conds, **map_kwargs)
            step.do_map()
            self.add_starting_points_from_step(step)

        self._dedup_harvested_starting_points()

    def add_starting_points_from_step(self, step: StepStrategy):
        """
        Grabs starting points from a step calc

        When mapping with a potential axis:
            - For stepping in a state variable (T or P), this is all the nodes of the step calc
            - For stepping in composition, this is all the 2 phase regions

        When mapping with only composition axes, this is all 2-phase and 3-phase regions,
        with a global min check to make sure these phase regions are truly the phases they say they are
        """
        # New starting points need to be processed by mapping before the next data retrieval
        self._mapping_complete = False
        if self.num_potential_condition > 0:
            self._add_starting_points_from_step_potential_axis(step)
        else:
            self._add_starting_points_from_step_composition_axes(step)

    def _add_starting_points_from_step_potential_axis(self, step: StepStrategy):
        """
        NOTE: Grabbing starting points is different for whether the axis variable on the step calculation is a state variable or not
            For stepping along a state variable where the composition is likely near an end point,
            the two-phase regions are usually too small to be resolved in the stepping resolution,
            thus, getting starting points from the node is more consistent. Only one starting point
            is added for each phase transition since for alpha->beta, the zpf line for alpha will end
            with the parent and the zpf line for beta will start with the node
            For stepping along a composition axis, just grab a single point from a zpf line.

            If we were to grab the nodes, there would be two nodes for every two-phase regions:
                alpha -> alpha + beta - Node for beginning of alpha + beta zpf line
                alpha + beta -> beta - Node for beginning of beta zpf line
        """
        # If stepping in a state variable, then grab all the nodes
        if map_utils.is_state_variable(step.axis_vars[0]):
            # Get all two-phase nodes. We set axis variable to None so that the node will find a good starting direction
            #  We force add nodes for positive and negative direction. This is in case the starting point ends up being in the middle
            #  of a zpf line (can happen for low solubility phases) so we want to step both in positive and negative direction
            # Nodes without a parent are starting points the step map force-added to recover
            #  from a failure (e.g. a failed node solve or a rejected metastable node). They can
            #  be the only record of a phase field whose bounding node solve failed (a narrow
            #  allotrope window straddled by the coarse edge step), so harvest them too. They
            #  are deduplicated against the parented nodes, which the step map guarantees are
            #  distinct phase-change events.
            harvested = [node for node in step.node_queue.nodes
                         if node.parent is not None and len(node.stable_composition_sets) == 2]
            for node in step.node_queue.nodes:
                if node.parent is None and len(node.stable_composition_sets) == 2:
                    if not any(node == other for other in harvested):
                        harvested.append(node)
            for node in harvested:
                _log.info(f"Adding node {node.fixed_phases}, {node.free_phases}, {node.global_conditions}")
                node.axis_var = None
                node.axis_direction = Direction.POSITIVE
                node.exit_hint = ExitHint.POINT_IS_EXIT
                node.skip_if_covered = True
                self.node_queue.add_node(node, True)

                alt_node = Node(node.global_conditions, node.chemical_potentials, node.fixed_composition_sets, node.free_composition_sets, node.parent)
                alt_node.axis_var = None
                alt_node.axis_direction = Direction.NEGATIVE
                alt_node.exit_hint = ExitHint.POINT_IS_EXIT
                alt_node.skip_if_covered = True
                self.node_queue.add_node(alt_node, True)

        # If stepping in non-state variable, then for all two-phase zpf lines, grab one point per
        # contiguous two-phase field to add as node
        else:
            # A step line can silently merge several distinct two-phase fields whose phases have
            # identical names (e.g. ordered/disordered variants of a partitioned phase): the
            # warm-started solver slides the same composition sets from one field into the next,
            # so no phase-change node separates them. With potential conditions fixed, tie-line
            # endpoint compositions are constant within each two-phase field, so a jump in either
            # endpoint between consecutive points marks a new field. Harvest one starting point
            # per contiguous segment so that every merged field gets seeded.
            segment_tol = 1e-3
            harvested_points = []
            for zpf_line in step.zpf_lines:
                if len(zpf_line.stable_phases) == 2:
                    segment_points = []
                    prev_comps = None
                    for point in zpf_line.points:
                        if len(point.stable_phases) != 2:
                            continue
                        comps = np.sort([np.asarray(cs.X, dtype=float) for cs in point.stable_composition_sets], axis=0)
                        if prev_comps is None or np.amax(np.abs(comps - prev_comps)) > segment_tol:
                            # A step line can jump out of a field and back into it, so
                            # dedup against points with the same tie-line already harvested
                            # (tie-line endpoints identify the field at fixed potentials)
                            if not any(point == other for other in harvested_points):
                                segment_points.append(point)
                                harvested_points.append(point)
                        prev_comps = comps

                    for new_point in segment_points:
                        _log.info(f"Adding point {new_point.fixed_phases}, {new_point.free_phases}, {new_point.global_conditions}")
                        node = self._create_node_from_point(new_point, None, None, Direction.POSITIVE, ExitHint.POINT_IS_EXIT)
                        node.skip_if_covered = True
                        self.node_queue.add_node(node, True)

                        node = self._create_node_from_point(new_point, None, None, Direction.NEGATIVE, ExitHint.POINT_IS_EXIT)
                        node.skip_if_covered = True
                        self.node_queue.add_node(node, True)

    def _add_starting_points_from_step_composition_axes(self, step: StepStrategy):
        for zpf_line in step.zpf_lines:
            if len(zpf_line.stable_phases) == 2 or len(zpf_line.stable_phases) == 3:
                num_phases = len(zpf_line.stable_phases)

                # Iterate through the zpf line until we find a point with the same number of phases as the
                # zpf line itself. This seems redundant, but it is because the first point of the zpf line
                # is likely going to be a node or a node parent, where the number of phases may differ by 1
                p_index = 0
                while len(zpf_line.points[p_index].stable_phases) != num_phases:
                    p_index += 1

                # If we do find a good point on the zpf line, then we can process it to be a starting point
                if len(zpf_line.points[p_index].stable_phases) == num_phases:
                    new_point = zpf_line.points[p_index]
                    if self.all_vars[-1] in new_point.global_conditions:
                        del new_point.global_conditions[self.all_vars[-1]]
                        new_point.global_conditions[self.axis_vars[0]] = new_point.get_property(self.axis_vars[0])
                    _log.info(f"Adding node {new_point.fixed_phases}, {new_point.free_phases}, {new_point.global_conditions}")

                    # New point will be make with all composition sets as free, since we set
                    # which CS to fix when finding an exit
                    free_point = map_utils._generate_point_with_free_cs(new_point)

                    cs_result = zeq._find_global_min_cs(free_point, system_info=self.system_info, pdens=self.GLOBAL_MIN_PDENS, tol=self.GLOBAL_MIN_TOL, num_candidates=self.GLOBAL_MIN_NUM_CANDIDATES)
                    if cs_result is None:
                        new_node = self._create_node_from_point(free_point, None, None, None)
                        self.node_queue.add_node(new_node)

    def _validate_custom_starting_point(self, point: Point, direction: Direction):
        """
        Modifies exit hint and direction based off number of composition sets
            Single phase -> cannot be added
            Two phase    -> point is exit, direction stays the same as input
            Three phase  -> normal exit finding strategy, no direction needed
        """
        if len(point.stable_composition_sets) <= 1:
            return None, None, "Single phase detected"
        elif len(point.stable_composition_sets) == 2:
            return ExitHint.POINT_IS_EXIT, direction, None
        elif len(point.stable_composition_sets) == 3:
            return ExitHint.NORMAL, None, None
        else:
            return None, None, "More than three phases detected"

    def _find_exits_from_node(self, node: Node):
        """
        A node has three exits, which are combinations of 2 CS in the node
        Since the node is found from one pair of CS, one of the exits are already accounted for, so
        practically, it's only 2 exits
            However, if there are multiple starting points, a node may be found from multiple zpf lines
            thus a node may only have 1 or even 0 exits - (does this mean some exits are repeated if a node is found twice?)
        """
        exits, exit_dirs = super()._find_exits_from_node(node)
        if node.exit_hint == ExitHint.POINT_IS_EXIT:
            return exits, exit_dirs

        for cs_1, cs_2 in itertools.combinations(node.stable_composition_sets, 2):
            new_conds = {key: node.get_property(key) for key, val in node.global_conditions.items()}

            # Create point with free composition sets (and ignore whether cs_1 or cs_2 is actually fixed)
            # This will be modified in _determine_start_direction
            candidate_point = Point.with_copy(new_conds, node.chemical_potentials, [], [cs_1, cs_2])

            if not node.has_point_been_encountered(candidate_point, False):
                _log.info(f"Found candidate exit: {candidate_point.fixed_phases}, {candidate_point.free_phases}, {candidate_point.global_conditions}")
                exits.append(candidate_point)
                # This function is only responsible for finding exit points
                # The _determine_start_direction will take the point and find the direction and axis variable
                exit_dirs.append(None)

        return exits, exit_dirs

    def _dedup_harvested_starting_points(self):
        """
        Removes queued harvested starting points that duplicate an earlier queued one:
        same set of stable phases, same starting direction, same tie-line (potential
        coordinates within a step and phase compositions matching)

        The different edge harvests can seed the same boundary more than once (e.g. a
        parentless recovery node from an x-edge temperature step a fraction of a step
        above a T-edge composition-step point on the same boundary). Each seed lands in
        the start region of the other's traced line, which the runtime coverage check
        deliberately exempts (see _point_on_existing_zpf_line), so these duplicates can
        only be removed before tracing starts. Both directions of the surviving
        representative are kept.
        """
        potential_axes = [av for av in self.axis_vars if map_utils.is_state_variable(av)]
        kept = []
        kept_keys = []
        for node in self.node_queue.nodes:
            if not getattr(node, "skip_if_covered", False):
                kept.append(node)
                continue
            phases = sorted(node.stable_phases)
            pot = np.array([np.squeeze(node.get_property(av)) / self.normalize_factor(av) for av in potential_axes], dtype=float)
            comps = np.sort([np.asarray(cs.X, dtype=float) for cs in node.stable_composition_sets], axis=0)
            duplicate = False
            for o_phases, o_dir, o_pot, o_comps in kept_keys:
                if (o_phases == phases and o_dir == node.axis_direction
                        and (len(pot) == 0 or np.amax(np.abs(o_pot - pot)) < 0.75)
                        and o_comps.shape == comps.shape and np.amax(np.abs(o_comps - comps)) < 0.01):
                    duplicate = True
                    break
            if duplicate:
                _log.info(f"Dropping duplicate harvested starting point {node.stable_phases}, {node.global_conditions}")
                continue
            kept.append(node)
            kept_keys.append((phases, node.axis_direction, pot, comps))
        self.node_queue.nodes = kept

    def _point_on_existing_zpf_line(self, point: Point):
        """
        Returns True if an existing zpf line already traces the tie-line that point lies
        on: a line with the same set of stable phases containing a point with the same
        potential-axis coordinates and the same phase compositions

        Position in condition space alone cannot decide coverage: distinct two-phase
        fields with the same phase pair flank a line compound at nearly identical
        conditions (only their tie-line endpoint compositions differ, by up to the
        compound-to-compound spacing), and conversely every global composition along a
        tie-line shares that tie-line, so the lever position (the composition condition)
        must be ignored. A tie-line is identified by its potential coordinates plus its
        phase compositions.

        Points within one normalized step of a line's own first point do not count as
        covering: a line merely starting at a position covers only the side it traced,
        and the opposite-direction sibling of a starting point (which sits at, or within
        a refined first step of, that first point) must still be allowed to trace the
        other side of the boundary.
        """
        point_phases = sorted(point.stable_phases)
        potential_axes = [av for av in self.axis_vars if map_utils.is_state_variable(av)]
        point_pot = np.array([np.squeeze(point.get_property(av)) / self.normalize_factor(av) for av in potential_axes], dtype=float)
        point_comps = np.sort([np.asarray(cs.X, dtype=float) for cs in point.stable_composition_sets], axis=0)
        for zpf_line in self.zpf_lines:
            if sorted(zpf_line.stable_phases) != point_phases:
                continue
            first_pos = np.array([np.squeeze(zpf_line.points[0].get_property(av)) / self.normalize_factor(av) for av in self.axis_vars], dtype=float)
            for line_point in zpf_line.points[1:]:
                line_pos = np.array([np.squeeze(line_point.get_property(av)) / self.normalize_factor(av) for av in self.axis_vars], dtype=float)
                # Start region of the line: does not cover the other direction
                if np.sqrt(np.sum((line_pos - first_pos)**2)) < 1.0:
                    continue
                if len(potential_axes) > 0:
                    line_pot = np.array([np.squeeze(line_point.get_property(av)) / self.normalize_factor(av) for av in potential_axes], dtype=float)
                    if np.amax(np.abs(line_pot - point_pot)) >= 0.75:
                        continue
                line_comps = np.sort([np.asarray(cs.X, dtype=float) for cs in line_point.stable_composition_sets], axis=0)
                if line_comps.shape == point_comps.shape and np.amax(np.abs(line_comps - point_comps)) < 0.01:
                    _log.info(f"Point {point.stable_phases}, {point.global_conditions} is already covered by zpf line {zpf_line.stable_phases} (same tie-line). Skipping.")
                    return True
        return False

    def _determine_start_direction(self, node: Node, exit_point: Point, proposed_direction: Direction):
        """
        For stepping, only one direction is possible from a node since we either step positive or negative

        If a direction cannot be found, then we force add a starting point just past the exit_point
        """
        # Skip exits that would re-trace an already-mapped tie-line, to avoid duplicate
        # boundary traces: automatically harvested starting points (skip_if_covered) and
        # exits from mapping-found or recovery nodes (ExitHint.NORMAL, e.g. re-found
        # invariants whose exits would re-trace covered fields). A two-phase starting
        # point added explicitly by a user (POINT_IS_EXIT without skip_if_covered) is
        # always honored
        if (getattr(node, "skip_if_covered", False) or node.exit_hint == ExitHint.NORMAL) and self._point_on_existing_zpf_line(exit_point):
            return None

        if self.num_potential_condition > 0:
            return self._determine_start_direction_potential_axis(node, exit_point, proposed_direction)
        else:
            return self._determine_start_direction_composition_axes(node, exit_point, proposed_direction)

    def _determine_start_direction_potential_axis(self, node: Node, exit_point: Point, proposed_direction: Direction):
        # If no proposed direction, then we test both directions
        if proposed_direction is None:
            directions = [Direction.POSITIVE, Direction.NEGATIVE]
        else:
            directions = [proposed_direction]

        # Sort exit point to fix composition set that varies the least and set axis variable with av1 where d(av1)/d(av2) > d(av2)/d(av1)
        norm = {av: self.normalize_factor(av) for av in self.axis_vars}
        exit_point, axis_var = _sort_point_potential_axis(exit_point, self.axis_vars, norm)
        for d in directions:
            dir_results = self._test_direction(exit_point, axis_var, d)
            if dir_results is not None:
                av_delta, other_av_delta = dir_results
                _log.info(f"Found direction: {axis_var, d, av_delta} for point {exit_point.fixed_phases}, {exit_point.free_phases}, {exit_point.global_conditions}")
                return exit_point, axis_var, d, av_delta

        return None

    def _determine_start_direction_composition_axes(self, node: Node, exit_point: Point, proposed_direction: Direction):
        # Sort exit point to fix composition set that varies the least
        der, exit_point, axis_var, normal = _sort_point_composition_axes(exit_point, self.axis_vars)

        free_cs = exit_point.free_composition_sets[0]
        # If node is invariant, then we can get the other cs to know which direction to step away from
        # It's possible that the node is not invariant (ex. as a starting point), in which case, we just set the norm_delta_dot to 1
        if len(node.stable_composition_sets) == 3:
            other_cs = [cs for cs in node.stable_composition_sets if not _eq_compset(cs, exit_point.fixed_composition_sets[0]) and not _eq_compset(cs, exit_point.free_composition_sets[0])][0]
            delta_var = _get_delta_cs_var(exit_point, [free_cs, other_cs], self.axis_vars)
            norm_delta_dot = np.dot(delta_var, normal)
        else:
            norm_delta_dot = 1


        if proposed_direction is None:
            # If the derivative of the axis variable is positive and the normal is the same direction as the delta
            # Or if derivative is negative and normal is opposite direction, then stepping is Positive direction
            # would be preferred. Same goes for the two other cases where Negative direction is preferred
            # We add the other direction just in case the first proposed direction fails
            if norm_delta_dot * der > 0:
                directions = [Direction.POSITIVE, Direction.NEGATIVE]
            else:
                directions = [Direction.NEGATIVE, Direction.POSITIVE]
        else:
            directions = [proposed_direction]

        for d in directions:
            dir_results = self._test_direction(exit_point, axis_var, d)
            if dir_results is not None:
                av_delta, other_av_delta = dir_results
                _log.info(f"Found direction: {axis_var, d, av_delta} for point {exit_point.fixed_phases}, {exit_point.free_phases}, {exit_point.global_conditions}")
                return exit_point, axis_var, d, av_delta

        self._add_starting_point_at_new_condition(exit_point, normal, Direction.POSITIVE if norm_delta_dot > 0 else Direction.NEGATIVE)

        return None

    def _add_starting_point_at_new_condition(self, point: Point, normal: list[float], direction: Direction):
        """
        If we made it here, then no direction has worked. This could be a case where
        the two CS of the exit point are stoichiometric, so there is no ZPF line leading
        to the next node
        So we take a step from the normal of the exit away from the third CS of the invariant
        and add a new starting point. We also add the current exit as a parent, so we don't
        search in that direction again
        """
        free_point = map_utils._generate_point_with_free_cs(point)
        copy_conds = copy.deepcopy(free_point.global_conditions)
        # Move by small amount (1e-3 seem to be a good value, too small, and we may fail to detect a possible third phase and too large, and we may step over a potential node)
        for av, norm_dir in zip(self.axis_vars, normal):
            copy_conds[av] += 1e-3*norm_dir*direction.value

        _log.info(f"Attemping to add point at {copy_conds}")
        new_point = point_from_equilibrium(self.dbf, self.components, self.phases, copy_conds, models=self.models, phase_record_factory=self.phase_records)
        if new_point is not None:
            # If new point is an invariant, then we add it as a starting point (point is added as a parent to prevent repeated exit calculations)
            # If new point is 2 phase, then we add it as a starting point the node being the exit
            _log.info(f"Adding starting point: {new_point.stable_phases}")
            success = False
            if len(new_point.stable_composition_sets) == 3:
                new_node = self._create_node_from_point(new_point, point, None, None)
                success = self.node_queue.add_node(new_node)
            elif len(new_point.stable_composition_sets) == 2:
                new_node = self._create_node_from_point(new_point, None, None, None, ExitHint.POINT_IS_EXIT)
                success = self.node_queue.add_node(new_node)
            else:
                _log.info("Point could not be starting point. Needs 2 or 3 phases")
            if not success:
                _log.info("Point has already been added")
        else:
            _log.info("Could not find a new node from conditions")

    def _process_new_node(self, zpf_line: ZPFLine, new_node: Node):
        """
        Post global min check after finding node to ensure it's not a metastable node

        If the node is metastable, the zpf line is kept (the points on it have already
        passed their own global min checks) but ended, and the node is not added
        """
        _log.info("Checking if new node is metastable")
        cs_result = zeq._find_global_min_cs(new_node, system_info=self.system_info, pdens=self.GLOBAL_MIN_PDENS, tol=self.GLOBAL_MIN_TOL, num_candidates=self.GLOBAL_MIN_NUM_CANDIDATES)
        if cs_result is not None:
            if not zeq._detect_degenerate_phase(new_node, cs_result[0]):
                cs_result = None
        if cs_result is None:
            _log.info("Global eq check on new node passed")
            super()._process_new_node(zpf_line, new_node)
        else:
            _log.info("Global eq check failed. New node is metastable. Ending current zpf line.")
            zpf_line.status = ZPFState.FAILED

    def get_invariant_data(self, x: v.StateVariable, y: v.StateVariable, global_x: bool = False, global_y: bool = False) -> list[PhaseRegionData]:
        """
        Create a dictionary of data for invariant plotting.

        Parameters
        ----------
        x : v.StateVariable
            The state variable to be used for the x-axis.
        y : v.StateVariable
            The state variable to be used for the y-axis.
        global_x : bool
            Whether variable x applies to the global system
        global_y : bool
            Whether variable y applies to the global system

        Returns
        -------
        list of PhaseRegionData, which stores of list of SinglePhaseData with
        x, y coordinates for each phase in the invariant reaction
        """
        self._ensure_mapped()
        if hasattr(x, 'phase_name') and x.phase_name is None:
            if not global_x:
                x = copy.deepcopy(x)
                x.phase_name = '*'

        if hasattr(y, 'phase_name') and y.phase_name is None:
            if not global_y:
                y = copy.deepcopy(y)
                y.phase_name = '*'

        invariant_data = []
        for node in self.node_queue.nodes:
            # Nodes in tieline mappings are always 3 composition sets
            if len(node.stable_composition_sets) == 3:
                node_phases = node.stable_phases_with_multiplicity
                data = []
                for p in node_phases:
                    x_var = _get_phase_specific_variable(p, x)
                    y_var = _get_phase_specific_variable(p, y)
                    x_data = to_display_units(node.get_property(x_var), node.stable_composition_sets, x_var)
                    y_data = to_display_units(node.get_property(y_var), node.stable_composition_sets, y_var)
                    data.append(SinglePhaseData(p, x_data, y_data))
                invariant_data.append(StrategyData(data))

        return invariant_data

    def get_tieline_data(self, x: v.StateVariable, y: v.StateVariable, global_x: bool = False, global_y: bool = False) -> list[PhaseRegionData]:
        """
        Create a dictionary of data for plotting ZPF lines.

        Parameters
        ----------
        x : v.StateVariable
            The state variable to be used for the x-axis.
        y : v.StateVariable
            The state variable to be used for the y-axis.
        global_x : bool
            Whether variable x applies to the global system
        global_y : bool
            Whether variable y applies to the global system

        Returns
        -------
        list of PhaseRegionData, which stores of list of SinglePhaseData with
        x, y coordinates of the phase boundary for each phase
        """
        self._ensure_mapped()
        if hasattr(x, 'phase_name') and x.phase_name is None:
            if not global_x:
                x = copy.deepcopy(x)
                x.phase_name = '*'

        if hasattr(y, 'phase_name') and y.phase_name is None:
            if not global_y:
                y = copy.deepcopy(y)
                y.phase_name = '*'

        zpf_data = []
        for zpf_line in self.zpf_lines:
            phases = zpf_line.stable_phases_with_multiplicity
            data = []
            for p in phases:
                x_data = zpf_line.get_var_list(_get_phase_specific_variable(p, x))
                y_data = zpf_line.get_var_list(_get_phase_specific_variable(p, y))
                data.append(SinglePhaseData(p, x_data, y_data))
            zpf_data.append(StrategyData(data))

        return zpf_data


BinaryStrategy = TielineStrategy
TernaryStrategy = TielineStrategy
