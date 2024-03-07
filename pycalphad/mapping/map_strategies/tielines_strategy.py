from pycalphad_mapping.map_strategies.general_strategy import GeneralStrategy
from pycalphad import Database, calculate, variables as v
from typing import List, Mapping
from pycalphad_mapping.primitives import STATEVARS, Point, Node, _get_global_value_for_var, _get_value_for_var, ExitHint
import numpy as np
import itertools

"""
Strategy for when tielines are in plane

This is a special case of the general strategy where we know the solution to finding the exits
    The only thing that changes here is the exit strategy where we know there will always be two exits from a node
        a+b -> a+b+y
            Exits are a+b and b+y (2 phases combinations of a+b+y that isn't the parent point)
        We also may switch which phase to fix to make sure we don't run into issues during stepping if a composition of a phase is restricted (like for ordered or stoichiometric phases)
"""

class TielineStrategy (GeneralStrategy):
    def __init__(self, database: Database, components : List, elements : List, phases: List, conditions: Mapping):
        super().__init__(database, components, elements, phases, conditions)
        self.tielines = True

    def find_exits(self, node):
        """
        Node exits for when tielines are in the plane of the phase diagram
        Always 2 - nodes will be 3 phase equilibrium, so exits will be each pair of the 3 phases, excluding the node we started from
        """
        node_exits = []
        # Exactly 2 exits
        # If tie-lines are in the plane, the node must be a three phase equilibrium, where exits are pairs of phases
        assert len(node.stable_composition_sets) == 3, f"Expected exactly three phases in equilibrium if tie-lines are in the plane. Got {len(node.stable_composition_sets)} with {node.stable_composition_sets}."
        self.log(f"FIND_EXIT:\tNode parent: {node.parent}")
        #Assume cs_1 as the fixed composition set and cs_2 as the free composition set
        for cs_1, cs_2 in itertools.combinations(node.stable_composition_sets, 2):
            cs_1, cs_2 = self._test_fixed_free_initial_comp_sets(cs_1, cs_2)

            #TODO: this code copying the conditions and re-setting the axis variables to the composition sets is used a lot everywhere
            new_conditions = {k:v for k,v in node.global_conditions.items()}
            #For state variables, it doesn't matter which composition set we take it from
            #For other variables, we want to grab it from the free composition set
            for av in new_conditions:
                if av in STATEVARS:
                    new_conditions[av] = _get_value_for_var(cs_1, av)
                else:
                    new_conditions[av] = _get_value_for_var(cs_2, av)
            candidate_point = Point.with_copy(new_conditions, [cs_1], [cs_2], node.metastable_composition_sets)
            # Use _{fixed/free}_composition_sets because we haven't set fixed vs. free yet.
            candidate_point._fixed_composition_sets[0].fixed = True
            candidate_point._fixed_composition_sets[0].NP = 0.0
            candidate_point._free_composition_sets[0].fixed = False
            candidate_point._free_composition_sets[0].NP = 1.0
            self.log(f"FIND_EXIT:\tCandidate (eq={candidate_point == node.parent}): {candidate_point}")
            #if candidate_point != node.parent or node.exit_hint == ExitHint.FORCE_ALL_EXITS:
            if not node.has_point_been_encountered(candidate_point, False) or node.exit_hint == ExitHint.FORCE_ALL_EXITS:
                # this is a new exit
                node_exits.append(candidate_point)
        #Since we narrow the exits down to avoid double calculating ZPF lines, we don't want to assert this anymore
        #assert (len(node_exits) == 2) or (len(node_exits) == 3 and node.exit_hint == ExitHint.FORCE_ALL_EXITS), f"FIND_EXIT:\tExpected 2 exits, but got {len(node_exits)}. Exits: {node_exits}. Got stable composition sets: {node.stable_composition_sets}"
        return node_exits

    def _test_fixed_free_initial_comp_sets(self, cs_1, cs_2):
        #TODO: this is ad-hoc, but set the free composition set to the phase with the least DOO
        #if the same DOO, then pick the composition set furthest away from the composition axes
        #Basically, we want the free phase to be the on that can vary the most
        #ALSO TODO: this is similar to the _sort_cs_by_what_to_fix, and this should just use that funciton
        p_cs1 = np.array([cs_1.dof[len(STATEVARS):]])
        p_cs2 = np.array([cs_2.dof[len(STATEVARS):]])
        state_vars = {str(k): _get_value_for_var(cs_1, k) for k in [v.T, v.P, v.N]}
        doo_cs1 = calculate(self._system_definition["dbf"], self._system_definition["comps"], [cs_1.phase_record.phase_name],
                            output="DOO", model = self._system_definition["models"], points = p_cs1, **state_vars)
        doo_cs2 = calculate(self._system_definition["dbf"], self._system_definition["comps"], [cs_2.phase_record.phase_name],
                            output="DOO", model = self._system_definition["models"], points = p_cs2, **state_vars)
        doo_cs1 = doo_cs1.DOO.values.ravel()[0]
        doo_cs2 = doo_cs2.DOO.values.ravel()[0]
        if (doo_cs1 == doo_cs2 or (doo_cs1 < 0.01 and doo_cs2 < 0.01)):
            prod_cs1 = np.prod(cs_1.X) * np.prod(1-np.array(cs_1.X))
            prod_cs2 = np.prod(cs_2.X) * np.prod(1-np.array(cs_2.X))
            if prod_cs2 < prod_cs1:
                cs_1, cs_2 = cs_2, cs_1
        else:
            if doo_cs2 > doo_cs1:
                cs_1, cs_2 = cs_2, cs_1
        return cs_1, cs_2

    def determine_start_direction(self, node: Node):
        general_dir_results = super().determine_start_direction(node)
        #If we failed to find a valid search direction, then swap the fixed and free phases and try again
        if general_dir_results is None:
            self.log("START_DIR:\tNo direction found, swapping composition sets")
            cs_free = node._free_composition_sets
            cs_fixed = node._fixed_composition_sets
            for cs in cs_free:
                cs.fixed = True
                cs.NP = 0
            for cs in cs_fixed:
                cs.fixed = False
                cs.NP = 1 / len(cs_fixed)
            node._free_composition_sets = cs_fixed
            node._fixed_composition_sets = cs_free
            for av in self.axis_vars:
                node.global_conditions[av] = _get_global_value_for_var(node, av)
            return super().determine_start_direction(node)
        else:
            return general_dir_results