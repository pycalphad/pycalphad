from typing import Union
import logging
import itertools
import copy

import numpy as np

from pycalphad import Database, variables as v

from pycalphad.mapping.primitives import ZPFLine, Node, Point, ExitHint, Direction, MIN_COMPOSITION
import pycalphad.mapping.zpf_equilibrium as zeq
import pycalphad.mapping.utils as map_utils
from pycalphad.mapping.strategy.strategy_base import MapStrategy
from pycalphad.mapping.strategy.step_strategy import StepStrategy

_log = logging.getLogger(__name__)

def _sort_point(point: Point, axis_vars: list[v.StateVariable], norm: dict[v.StateVariable, float]):
    """
    Given a point in a binary system with 2 free phases, get derivative at both composition sets to
    test with CS to fix and which direction to start
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

class BinaryStrategy(MapStrategy):
    def __init__(self, dbf: Database, components: list[str], phases: list[str], conditions: dict[v.StateVariable, Union[float, tuple[float]]], **kwargs):
        super().__init__(dbf, components, phases, conditions, **kwargs)

    def initialize(self):
        """
        Searches axis limits to find starting points

        Here, we do a step mapping along the axis bounds and grab all the nodes
        The nodes of a step map is distinguished from starting points in that they have a parent
        """
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
                map_kwargs = self._constant_kwargs()
                step = StepStrategy(self.dbf, self.components, self.phases, conds, **map_kwargs)
                step.initialize()
                step.do_map()
                self._add_starting_points_from_step(step)

    def _add_starting_points_from_step(self, step: StepStrategy):
        """
        Grabs starting points from a step calc
            For stepping in a state variable (T or P), this is all the nodes of the step calc
            For stepping in composition, this is all the 2 phase regions
        
        NOTE: Grabbing starting points is different for whether the axis variable on the step calculation
              is a state variable or not
              
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
            # Get all nodes that has a parent. We set axis variable to None so that the node will find a good starting direction
            #  We force add nodes for positive and negative direction. This is in case the starting point ends up being in the middle
            #  of a zpf line (can happen for low solubility phases) so we want to step both in positive and negative direction
            for node in step.node_queue.nodes:
                if node.parent is not None and len(node.stable_composition_sets) == 2:
                    _log.info(f"Adding node {node.fixed_phases}, {node.free_phases}, {node.global_conditions}")
                    node.axis_var = None
                    node.axis_direction = Direction.POSITIVE
                    node.exit_hint = ExitHint.POINT_IS_EXIT
                    self.node_queue.add_node(node, True)

                    alt_node = Node(node.global_conditions, node.chemical_potentials, node.fixed_composition_sets, node.free_composition_sets, node.parent)
                    alt_node.axis_var = None
                    alt_node.axis_direction = Direction.NEGATIVE
                    alt_node.exit_hint = ExitHint.POINT_IS_EXIT
                    self.node_queue.add_node(alt_node, True)

        # If stepping in non-state variable, then for all two-phase zpf lines, grab one point to add as node
        else:
            for zpf_line in step.zpf_lines:
                if len(zpf_line.stable_phases) == 2:
                    p_index = 0
                    while len(zpf_line.points[p_index].stable_phases) != 2:
                        p_index += 1

                    if len(zpf_line.points[p_index].stable_phases) == 2:
                        new_point = zpf_line.points[p_index]
                        _log.info(f"Adding point {new_point.fixed_phases}, {new_point.free_phases}, {new_point.global_conditions}")
                        node = self._create_node_from_point(new_point, None, None, Direction.POSITIVE, ExitHint.POINT_IS_EXIT)
                        self.node_queue.add_node(node, True)

                        node = self._create_node_from_point(new_point, None, None, Direction.NEGATIVE, ExitHint.POINT_IS_EXIT)
                        self.node_queue.add_node(node, True)

    def _find_exits_from_node(self, node: Node):
        """
        A node on for a binary system has three exits, which are combinations of 2 CS in the node
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

            if not node.has_point_been_encountered(candidate_point, False) or node.exit_hint == ExitHint.FORCE_ALL_EXITS:
                _log.info(f"Found candidate exit: {candidate_point.fixed_phases}, {candidate_point.free_phases}, {candidate_point.global_conditions}")
                exits.append(candidate_point)
                # This function is only responsible for finding exit points
                # The _determine_start_direction will take the point and find the direction and axis variable
                exit_dirs.append(None)

        return exits, exit_dirs

    def _determine_start_direction(self, node: Node, exit_point: Point, proposed_direction: Direction):
        """
        For stepping, only one direction is possible from a node since we either step positive or negative

        If a direction cannot be found, then we force add a starting point just past the exit_point
        """
        # If no proposed direction, then we test both directions
        if proposed_direction is None:
            directions = [Direction.POSITIVE, Direction.NEGATIVE]
        else:
            directions = [proposed_direction]

        # Sort exit point to fix composition set that varies the least and set axis variable with av1 where d(av1)/d(av2) > d(av2)/d(av1)
        norm = {av: self.normalize_factor(av) for av in self.axis_vars}
        exit_point, axis_var = _sort_point(exit_point, self.axis_vars, norm)
        for d in directions:
            dir_results = self._test_direction(exit_point, axis_var, d)
            if dir_results is not None:
                av_delta, other_av_delta = dir_results
                _log.info(f"Found direction: {axis_var, d, av_delta} for point {exit_point.fixed_phases}, {exit_point.free_phases}, {exit_point.global_conditions}")
                return exit_point, axis_var, d, av_delta

        return None
    
    def _process_new_node(self, zpf_line: ZPFLine, new_node: Node):
        """
        Post global eq check after finding node to ensure it's not a metastable node
        """
        _log.info("Checking if new node is metastable")
        cs_result = zeq._find_global_min_cs(new_node, system_info=self.system_info, pdens=self.GLOBAL_MIN_PDENS, tol=self.GLOBAL_MIN_TOL, num_candidates=self.GLOBAL_MIN_NUM_CANDIDATES)
        if cs_result is None:
            _log.info("Global eq check on new node passed")
            super()._process_new_node(zpf_line, new_node)
        else:
            _log.info("Global eq check failed. New node is metastable. Removing current zpf line.")
            self.zpf_lines.pop(-1)

