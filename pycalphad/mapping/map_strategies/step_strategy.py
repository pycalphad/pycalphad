from pycalphad.mapping.map_strategies.strategy_base import MapStrategy
from pycalphad import Database, variables as v
from typing import List, Mapping
from pycalphad.mapping.primitives import ZPFLine, Point, Node, Direction, ExitHint
import itertools
import numpy as np

"""
Strategy for stepping
"""

class StepStrategy (MapStrategy):
    def __init__(self, database: Database, components : List, elements: List, phases: List, conditions: Mapping):
        super().__init__(database, components, elements, phases, conditions)
        self.stepping = True
        self.curr_direction = None

    def add_nodes_from_point(self, point: Point):
        self.node_queue.add_node(*self.convert_point_to_node(point, self.axis_vars[0], Direction.POSITIVE, ExitHint.NO_EXITS))
        self.node_queue.add_node(*self.convert_point_to_node(point, self.axis_vars[0], Direction.NEGATIVE, ExitHint.NO_EXITS))

    def find_exits(self, node):
        """
        Node exit for stepping
        For invariant points, exists will be all combinations of p-1 phases except the parent phase
        For starting point (no zpf lines yet), 2 exits for positive and negative directions
        For general case, 1 exit
        """
        num_comps, num_stable_phases, node_is_invariant = self._get_exit_info_from_node(node)
        node_exits = []
        self.curr_direction = node.axis_direction       #Temporary store the axis direction of the node so we can use that as the only direction when testing for candidate directions
        # Exactly 1 exit
        num_node_stable_compsets = len(node.stable_composition_sets)
        num_parent_stable_compsets = len(node.parent.stable_composition_sets)
        if num_node_stable_compsets - num_parent_stable_compsets == 1:
            # If the node has more phases than it's parent, a new phase is appearing.
            # By F = N + 2 - P - C, three phase regions on ternaries are also considered invariant
            #   but we want to treat them as normal phase boundaries when stepping in composition
            #   so we'll ignore invariants for the ternary case
            if node_is_invariant and self.num_potential_conditions > 0:
                # In order to add the new phase, one phase must be removed
                # We can check what phases are stable at the next by looking at all combinations of p-1 phases
                #   and set up a matrix of sum(NP^alpha x_i^alpha) = X_i and test whether all NP^alpha is positive
                set_parent_stable_compsets = set(node.parent.stable_composition_sets)
                assert len(set_parent_stable_compsets - set(node.stable_composition_sets)) == 0, f"All parent composition sets must be contained in the node. These compositions sets: {set_parent_stable_compsets - set(node.stable_composition_sets)} are not contained in the node stable composition sets: {node.stable_composition_sets}"
                for trial_stable_compsets in itertools.combinations(node.stable_composition_sets, num_stable_phases - 1):
                    if set(trial_stable_compsets) == set_parent_stable_compsets:
                        # Don't re-add the parent compsets
                        continue
                    phase_X_matrix = np.array([np.array(cs.X) for cs in trial_stable_compsets])
                    curr_comp = -1*np.ones(len(self.elements)-1)
                    sorted_el = sorted([el for el in self.elements if el != "VA"])
                    for el in sorted_el:
                        if v.X(el) in node.global_conditions:
                            curr_comp[sorted_el.index(el)] = node.global_conditions[v.X(el)]
                    curr_comp[curr_comp == -1] = -1*np.sum(curr_comp)     #This already accounts for the -1
                    phase_NP = np.matmul(np.linalg.inv(phase_X_matrix).T, np.array([curr_comp]).T)
                    if all(phase_NP.flatten() > 0):
                        candidate_point = Point.with_copy(node.global_conditions, [], list(trial_stable_compsets), node.metastable_composition_sets)
                        phase_frac_sum = 0
                        previous_fixed_cs = []
                        for cs in candidate_point.stable_composition_sets:
                            if cs.fixed:
                                previous_fixed_cs.append(cs)
                            cs.fixed = False
                            phase_frac_sum += cs.NP
                        if phase_frac_sum < 1:
                            for fixed_cs in previous_fixed_cs:
                                fixed_cs.NP = (1 - phase_frac_sum) / len(previous_fixed_cs)
                        node_exits.append(candidate_point)
                        break
            else:
                # The new exit should have all the stable phases in the node, but as free phases
                candidate_point = Point.with_copy(node.global_conditions, [], node.stable_composition_sets, node.metastable_composition_sets)
                # set all fixed phases to be free phases
                for cs in candidate_point.stable_composition_sets:
                    cs.fixed = False
                node_exits.append(candidate_point)
        elif num_node_stable_compsets == num_parent_stable_compsets:
            # If the node has the same number of phases as it's parent, then the phase
            # that is fixed to zero amount is disappearing and should be removed.
            MINIMUM_PHASE_AMOUNT = 1e-15
            compsets_to_keep = [cs for cs in node.stable_composition_sets if cs.NP > MINIMUM_PHASE_AMOUNT]
            assert len(compsets_to_keep) == (num_node_stable_compsets - 1), f"Found multiple nodes that are being destabilized {set(node.stable_composition_sets) - set(compsets_to_keep)}"
            # TODO:
            candidate_point = Point.with_copy(node.global_conditions, [], compsets_to_keep, node.metastable_composition_sets)
            # set all fixed phases to be free phases
            for cs in candidate_point.stable_composition_sets:
                cs.fixed = False
            node_exits.append(candidate_point)
        else:
            msg = (
                f"Node must have the same number of stable composition sets or one "
                f"more than its parent point. Got {num_node_stable_compsets} in the "
                f"node and {num_parent_stable_compsets} in its parent with "
                f"Node: {node} and its parent: {node.parent}."
            )
            raise ValueError(msg)
        return node_exits

    def determine_start_direction(self, node: Node):
        MIN_DELTA_RATIO = 0.02

        if self.curr_direction is None:
            directions = [Direction.POSITIVE, Direction.NEGATIVE]
        else:
            directions = [self.curr_direction]
        possible_directions = []
        for d in directions:
            valid, new_dir = self.test_direction(node, self.axis_vars[0], d, MIN_DELTA_RATIO)
            if valid:
                possible_directions.append(new_dir)

        if len(possible_directions) > 0:
            return self.find_best_direction(node, possible_directions)
        else:
            return None

    def find_best_direction(self, node: Node, possible_directions):
        return (possible_directions[0][0], possible_directions[0][1], possible_directions[0][2])

    def test_swap_axis(self, zpfline: ZPFLine):
        return
        if len(zpfline.points) < 2:
            zpfline.current_delta = self.axis_delta[zpfline.axis_var]