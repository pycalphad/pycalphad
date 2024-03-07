from pycalphad_mapping.map_strategies.strategy_base import MapStrategy
from pycalphad_mapping.map_strategies.step_strategy import StepStrategy
from pycalphad import Database, variables as v
from typing import List, Mapping
from pycalphad_mapping.primitives import ZPFLine, Point, Node, _get_global_value_for_var, _get_value_for_var, ExitHint, Direction
import numpy as np
import itertools
from pycalphad_mapping.primitives import STATEVARS
from pycalphad.core.constants import COMP_DIFFERENCE_TOL

"""
General strategy, used for isopleths
"""

class GeneralStrategy(MapStrategy):
    def __init__(self, database: Database, components : List, elements : List, phases: List, conditions: Mapping):
        super().__init__(database, components, elements, phases, conditions)
        self.tielines = False

    def add_nodes_from_point(self, point: Point):
        if v.T in self.axis_vars:
            curr_var = v.T
        elif v.P in self.axis_vars:
            curr_var = v.P
        else:
            curr_var = self.axis_vars[0]
        step_conditions = {k:v for k,v in point.global_conditions.items()}
        step_conditions[curr_var] = (self.axis_lims[curr_var][0], self.axis_lims[curr_var][1], self.axis_delta[curr_var])
        step_mapper = StepStrategy(self._system_definition["dbf"], self._system_definition["comps"], self.elements, self._system_definition["phases"], step_conditions)
        step_mapper.add_nodes_from_point(point)
        step_mapper.do_map()
        for zpfline in step_mapper.zpf_lines:
            if zpfline.num_fixed_phases() + zpfline.num_free_phases() > 1:
                # TODO: the starting and end points of a zpf line will be whenever there's a phase change and can be considered a node
                # Should we test for all nodes or just one end of the zpf line
                end_points = [0, -1]  # This will double count what ever phase regions the stepping crosses
                # end_points = [-1]  # This won't double count the phase regions, but I'm still not sure how good it works
                # Test how large the zpf line is in the current axis variable, if it's very small (< variable delta), then switch axis
                selected_zpf_var = curr_var
                zpf_range = np.abs(_get_global_value_for_var(zpfline.points[0], curr_var) - _get_global_value_for_var(zpfline.points[-1], curr_var))
                if zpf_range < self.axis_delta[curr_var]:
                    selected_zpf_var = self.axis_vars[int(1-self.axis_vars.index(curr_var))]
                for ind in end_points:
                    has_fixed_cs = False
                    for cs in zpfline.points[ind].stable_composition_sets:
                        if cs.NP == 0:
                            cs.fixed = True
                            has_fixed_cs = True
                    self.log("START_POINT:\t", zpfline.points[ind], [cs.NP for cs in zpfline.points[ind].stable_composition_sets], has_fixed_cs, isinstance(zpfline.points[ind], Node))
                    if has_fixed_cs:
                        if isinstance(zpfline.points[ind], Point):
                            self.node_queue.add_node(*self.convert_point_to_node(zpfline.points[ind], selected_zpf_var, Direction.POSITIVE, ExitHint.NO_EXITS))
                            self.node_queue.add_node(*self.convert_point_to_node(zpfline.points[ind], selected_zpf_var, Direction.NEGATIVE, ExitHint.NO_EXITS))
                        else:
                            self.node_queue.add_node(*self.copy_node_with_new_start_dir(zpfline.points[ind], selected_zpf_var, Direction.POSITIVE, ExitHint.NO_EXITS))
                            self.node_queue.add_node(*self.copy_node_with_new_start_dir(zpfline.points[ind], selected_zpf_var, Direction.NEGATIVE, ExitHint.NO_EXITS))

    def find_exits(self, node):
        """
        Node exits for when tielines are in the plane of the phase diagram
        Always 2 - nodes will be 3 phase equilibrium, so exits will be each pair of the 3 phases, excluding the node we started from
        """
        num_comps, num_stable_phases, node_is_invariant = self._get_exit_info_from_node(node)
        node_exits = []
        # Two cases
        if node_is_invariant:   #TODO: do we need to check if isothermal+isobaric like we do for stepping?
            # Maximum number of exits if 2*p - a ZPF line entering and exiting the node for each phase
            # We test the exits by going through combinations of n-1 phases (assume a and b are fixed or forbidden)
            #   Create a matrix of NP*x = X and test if NP is positive for all phases
            #   If this is true, then we create two exits, one with (n-1) (a) and (n-1) (b)
            #   Rather than keeping track of the two missing phases, we'll create combinations of all composition sets
            #       Then the nth phase will be fixed to 0 and n+1 th phase will be ignored if the candidate point passes
            #       This leads to some double calculations, but it's the linear system we solve is pretty small, so whatever
            for trial_stable_compsets in itertools.permutations(node.stable_composition_sets, num_stable_phases):
                phase_X_matrix = np.array([np.array(cs.X) for cs in trial_stable_compsets[:-2]])
                fixed_var = [av for av in node.global_conditions if (av != v.T and av != v.P and av not in self.axis_vars)]
                phase_X_matrix = np.zeros((len(fixed_var), len(trial_stable_compsets[:-2])))
                b = np.zeros((len(fixed_var),1))
                for i in range(len(fixed_var)):
                    for j in range(len(trial_stable_compsets[:-2])):
                        phase_X_matrix[i,j] = _get_value_for_var(trial_stable_compsets[j], fixed_var[i])
                    b[i,0] = node.global_conditions[fixed_var[i]]
                if np.linalg.matrix_rank(phase_X_matrix) != phase_X_matrix.shape[0]:
                    continue
                phase_NP = np.matmul(np.linalg.inv(phase_X_matrix), b).flatten()
                if all(phase_NP > 0):
                    candidate_point = Point.with_copy(node.global_conditions, [trial_stable_compsets[-2]], [cs for cs in trial_stable_compsets[:-2]], node.metastable_composition_sets)
                    for cs in candidate_point._fixed_composition_sets:
                        cs.fixed = True
                        cs.NP = 0
                    i = 0
                    for cs in candidate_point._free_composition_sets:
                        cs.fixed = False
                        cs.NP = phase_NP[i]
                        i += 1
                    for av in self.axis_vars:
                        candidate_point.global_conditions[av] = _get_global_value_for_var(candidate_point, av)
                    #isParent = candidate_point.compare_consider_fixed_cs(node.parent)
                    added = any(candidate_point.compare_consider_fixed_cs(cs) for cs in node_exits)
                    exit_has_been_encountered = node.has_point_been_encountered(candidate_point, True)
                    #if not isParent and not added and not exit_has_been_encountered:
                    if not added and not exit_has_been_encountered:
                        node_exits.append(candidate_point)

        else:
            # For non-invariant cases, the node will have two fixed phases
            # There will always be three exits
            # At any intersections, there are 4 regions, two regions opposite with the same phase and two regions opposite that differ by one
            # For phases a and b that are fixed at the intersection and P being any number of phases, the 4 zpf lines will be
            #    (P, b) (a)     Crosses P+a+b -> P+b
            #    (P, a) (b)     Crosses P+a+b -> P+a
            #    (P)    (a)     Crosses P+a -> P
            #    (P)    (b)     Crosses P+b -> P
            assert len(node.fixed_composition_sets) == 2
            for i in range(2):
                # Test for (P) (a)
                candidate_point = Point.with_copy(node.global_conditions, [node.fixed_composition_sets[i]], [cs for cs in node.free_composition_sets], node.metastable_composition_sets)
                self.log("CANDIDATE: ", candidate_point)
                # if not candidate_point.compare_consider_fixed_cs(node.parent) and not node.has_point_been_encountered(candidate_point, True):
                if not node.has_point_been_encountered(candidate_point, True):
                    node_exits.append(candidate_point)
                else:
                    self.log("Same as parent")
                # Test for (P, b) (a)
                candidate_point = Point.with_copy(node.global_conditions, [node.fixed_composition_sets[i]], [cs for cs in node.free_composition_sets] + [node.fixed_composition_sets[1-i]], node.metastable_composition_sets)
                for cs in candidate_point._free_composition_sets:
                    cs.fixed = False
                self.log("CANDIDATE: ", candidate_point)
                # if not candidate_point.compare_consider_fixed_cs(node.parent) and not node.has_point_been_encountered(candidate_point, True):
                if not node.has_point_been_encountered(candidate_point, True):
                    node_exits.append(candidate_point)
                else:
                    self.log("Same as parent")
        return node_exits

    def determine_start_direction(self, node: Node):
        # Check if node is using two phases with the same composition
        if not self.check_similar_phase_compositions(node, COMP_DIFFERENCE_TOL):
            self.log("DIRECTION:\tInvalid node to step condition")
            return None

        directions = [Direction.POSITIVE, Direction.NEGATIVE]
        possible_directions = []

        # Split axis variables to state vars and composition
        free_state_var = [av for av in self.axis_vars if av in STATEVARS]
        free_comp_var = [av for av in self.axis_vars if av not in STATEVARS]

        # Test state variables first
        for av in free_state_var:
            for d in directions:
                valid, new_dir = self.test_direction(node, av, d, 1)
                if valid:
                    possible_directions.append(new_dir)

        # Get best direction for state variables and get normalized delta (normalized to axis stepping)
        # If delta is close to 1, then zpf line is near vertical, in this case, don't bother with checking
        #    composition axis, since it could likely fail
        if len(possible_directions) > 0:
            best_state_dir, delta = self.find_best_direction(node, possible_directions, True)
            if delta < 1.05:
                return best_state_dir

        for av in free_comp_var:
            for d in directions:
                valid, new_dir = self.test_direction(node, av, d, 1)
                if valid:
                    possible_directions.append(new_dir)

        if len(possible_directions) > 0:
            return self.find_best_direction(node, possible_directions)
        else:
            return None

    def find_best_direction(self, node: Node, possible_directions, return_delta = False):
        """
        Multiple directions may be possible when exiting a node, so we want to figure out which exit will give us the best resolution line

        For tielines in plane, we want to also check which composition set varied the least and set that as fixed to 0 and other phase as 1, then the global condition should be adjusted to reflect that
        For isopleths, we want to keep the same fixed phase since only the fixed phase is guaranteed to be on the isopleth projection
        """
        max_delta = 1e6
        max_index = 0
        lowest_phase_change_index = 0
        p_index = 0
        for p in possible_directions:
            phase_delta = []
            # We only make it here if the set of stable phases doesn't change during initial stepping, so we can assume the same order of composition sets
            new_cs = p[3].stable_composition_sets
            prev_cs = p[4].stable_composition_sets
            for i in range(len(new_cs)):
                phase_delta.append(np.array([(_get_value_for_var(new_cs[i], av) - _get_value_for_var(prev_cs[i], av))/self.normalize_factor(av) for av in self.axis_vars]))
            delta_mag = [np.sqrt(np.sum(pdelta**2)) for pdelta in phase_delta]
            if np.amax(delta_mag) < max_delta:
                max_delta = np.amax(delta_mag)
                max_index = p_index
                lowest_phase_change_index = np.argmin(delta_mag)
            p_index += 1

        """
        if self.tielines:
            csIndex = 0
            for cs in node.stable_composition_sets:
                cs.fixed = False
                cs.NP = 1 / (len(node.stable_composition_sets)-1)
                if csIndex == lowest_phase_change_index:
                    cs.fixed = True
                    cs.NP = 0
                csIndex += 1
            csList = [cs for cs in node.stable_composition_sets]
            node._free_composition_sets = [cs for cs in csList if cs.fixed]
            node._fixed_composition_sets = [cs for cs in csList if not cs.fixed]
            for av in self.axis_vars:
                node.global_conditions[av] = _get_global_value_for_var(node, av)
        """

        self.log("DIRECTION:\tFound best direction: ", (possible_directions[max_index][0], possible_directions[max_index][1], possible_directions[max_index][2]))
        if return_delta:
            return (possible_directions[max_index][0], possible_directions[max_index][1], possible_directions[max_index][2]), max_delta
        else:
            return (possible_directions[max_index][0], possible_directions[max_index][1], possible_directions[max_index][2])

    def test_swap_axis(self, zpfline: ZPFLine):
        """
        If only 1 point in ZPF line, then we can take from the starting zpf line axis and direction
        If there are at least 2 points, then we take the slope of the last line and select the direction that gave the largest change

        Since we normalize the change in axis variable to the axis delta, we should be limiting the maximum possible change to 1*axis_delta
        """
        if len(zpfline.points) > 1:
            # Normalize to the axis limits
            delta_vs = []
            for av in self.axis_vars:
                dv = _get_global_value_for_var(zpfline.points[-1], av) - _get_global_value_for_var(zpfline.points[-2], av)
                av_range = self.normalize_factor(av)
                delta_vs.append(dv/av_range)
            # Go in the direction of the largest increment
            index = np.argmax(np.abs(delta_vs))
            direction = Direction.POSITIVE if delta_vs[index] > 0 else Direction.NEGATIVE
            if zpfline.axis_var != self.axis_vars[index]:
                zpfline.axis_var = self.axis_vars[index]
                zpfline.axis_direction = direction
                zpfline.current_delta = self.axis_delta[zpfline.axis_var]
                self.log("SWITCH_AXIS:\tSwitching axis: ", zpfline.axis_var, zpfline.axis_direction, zpfline.current_delta)

            # When starting a zpf line from a starting point, we set the axis increment to the minimum to account for very small phase regions (e.g. FCC/BCC for Fe-X systems)
            # For these cases, if we didn't have to switch axis, then we'll assume we can go back to the user-defined increment
            # TODO: this can cause issues if there is a convergence failure on the 3rd point in the zpf line
            else:
                if len(zpfline.points) == 2 and zpfline.current_delta == self.axis_delta[zpfline.axis_var]*self.MIN_DELTA_RATIO:
                    zpfline.current_delta = self.axis_delta[zpfline.axis_var]
