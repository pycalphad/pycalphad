from typing import Union
import logging
import itertools
import copy

import numpy as np

from pycalphad import Database, variables as v
import pycalphad.core.constants as PYC_CONSTANTS

from pycalphad.mapping.primitives import STATEVARS, ZPFLine, Node, Point, ExitHint, Direction, ZPFState, _get_phase_specific_variable
import pycalphad.mapping.utils as map_utils
from pycalphad.mapping.strategy.strategy_base import MapStrategy

_log = logging.getLogger(__name__)

class StepStrategy(MapStrategy):
    def __init__(self, dbf: Database, components: list[str], phases: list[str], conditions: dict[v.StateVariable, Union[float, tuple[float]]], **kwargs):
        super().__init__(dbf, components, phases, conditions, **kwargs)

    def initialize(self):
        """
        Adds initial starting point in middle of free condition
        """
        av = self.axis_vars[0]
        mid_val = (self.axis_lims[av][0] + self.axis_lims[av][1]) / 2
        mid_conds = copy.deepcopy(self.conditions)
        mid_conds[av] = mid_val
        self.add_nodes_from_conditions(mid_conds, None, None, True)

    def _find_exits_from_node(self, node: Node):
        """
        Apart from the default POINT_IS_EXIT condition from strategy base, there are three possibilities

        1. Node has more phases than parent and node is an invariant and we step in potential condition
            Test all n-1 combinations of the set of phases (excluding parent) to
            find the set will all phase fractions are positive when solving
            against the current composition
            Note: we shouldn"t have to account for the number of phases < number of compositions
                    since the matrix will be (comps x phases) and should always be full rank even
                    if comps > phases

        2. Node has more phases than parent and we step in non-potential condition
            Create exit with all phases stable

        3. Node has the same number of phases as the parent
            Remove phase with 0 phase fraction and create exit with remaining phases

        For stepping, all phases in an exit are free
        """
        exits, exit_dirs = super()._find_exits_from_node(node)
        if node.exit_hint == ExitHint.POINT_IS_EXIT:
            return exits, exit_dirs
        
        num_node_cs = len(node.stable_composition_sets)
        num_parent_cs = len(node.parent.stable_composition_sets)
        is_pot_cond = self.axis_vars[0] in STATEVARS
        is_invariant = map_utils.degrees_of_freedom(node, self.components, self.num_potential_condition) == 0

        if num_node_cs == num_parent_cs + 1:
            #Node has more phases than parent
            if is_pot_cond and is_invariant:
                #Potential condition, create matrix for each n-1 set of phases
                node_cs_set = set(node.stable_composition_sets)
                parent_cs_set = set(node.parent.stable_composition_sets)

                #Make sure parent cs if a subset of node cs
                if len(parent_cs_set - node_cs_set) != 0:
                    return exits, exit_dirs
                
                #Test all n-1 set of phases excluding parent set
                for trial_stable_cs in itertools.combinations(node.stable_composition_sets, num_node_cs - 1):
                    if set(trial_stable_cs) == parent_cs_set:
                        continue
                    # comps x phases
                    phase_matrix = np.array([cs.X for cs in trial_stable_cs]).T
                    # composition list
                    global_comps = [node.get_property(v.X(e)) for e in self.elements]
                    #phase fraction
                    phase_NP = np.linalg.lstsq(phase_matrix, global_comps, rcond=None)[0].flatten()
                    if all(phase_NP > 0):
                        candidate_point = Point.with_copy(node.global_conditions, node.chemical_potentials, [], list(trial_stable_cs))
                        #Since we have the phase fraction, we can update the cs with them
                        for cs, ph_np in zip(candidate_point.stable_composition_sets, phase_NP):
                            cs.fixed = False
                            map_utils.update_cs_phase_frac(cs, ph_np)
                        exits.append(candidate_point)
                        exit_dirs.append(node.axis_direction)
                        return exits, exit_dirs
            
            else:
                #Not potential condition, create exit with all phases stable and free
                candidate_point = Point.with_copy(node.global_conditions, node.chemical_potentials, [], node.stable_composition_sets)
                for cs in candidate_point.stable_composition_sets:
                    cs.fixed = False
                #Add candidate point with the same direction as the node
                exits.append(candidate_point)
                exit_dirs.append(node.axis_direction)
                return exits, exit_dirs

        elif num_node_cs == num_parent_cs:
            #Number of phases are the same, remove the 0 phase
            cs_to_keep = [cs for cs in node.stable_composition_sets if cs.NP > PYC_CONSTANTS.MIN_PHASE_FRACTION]
            #If there are more than 1 zero phase, then return the empty exits, here, a new starting point should be generated
            if len(cs_to_keep) < num_node_cs - 1:
                return exits, exit_dirs
            candidate_point = Point.with_copy(node.global_conditions, node.chemical_potentials, [], cs_to_keep)
            for cs in cs_to_keep:
                cs.fixed = False
            exits.append(candidate_point)
            exit_dirs.append(node.axis_direction)
            return exits, exit_dirs
        
        return exits, exit_dirs
        
        
    def _determine_start_direction(self, node: Node, exit_point: Point, proposed_direction: Direction):
        """
        For stepping, only one direction is possible from a node since we either step positive or negative

        If a direction cannot be found, then we force add a starting point just past the exit_point
        """
        axis_deltas = self._test_direction(exit_point, self.axis_vars[0], proposed_direction)
        if axis_deltas is None:
            #Test direction failed, so add a new starting point
            self._add_starting_point_at_last_condition(exit_point.global_conditions, proposed_direction)
            return None
        else:
            #Return axis variable, proposed direction and axis delta
            #  For the most point, this seems pointless since we of course know the
            #  axis variable and direction when stepping, but this is mainly for
            #  compatibility with the tielines and isopleth strategies
            av_delta, other_av_delta = axis_deltas
            return exit_point, self.axis_vars[0], proposed_direction, av_delta

    def _attempt_to_add_point(self, zpf_line: ZPFLine, step_results: tuple[Point, list]):
        """
        If a point was not added because the zpf line failed, then we force a starting point just past the
        end of the zpf line
        """
        super()._attempt_to_add_point(zpf_line, step_results)
        if zpf_line.status == ZPFState.FAILED:
            #ZPF line failed, so add a new starting point
            self._add_starting_point_at_last_condition(zpf_line.points[-1].global_conditions, zpf_line.axis_direction)
    
    def _add_starting_point_at_last_condition(self, conditions: dict[v.StateVariable, float], axis_dir: Direction):
        """
        Checks if the point is at the axis limits
        If not, then create a new starting point with the adjusted conditions
        """
        av = self.axis_vars[0]
        # Since we add the new starting point at MIN_DELTA_RATIO*axis_delta,
        # make sure that the new conditions won"t be past the axis limits
        new_delta = self.MIN_DELTA_RATIO * self.axis_delta[av]

        not_at_axis_lims = True
        new_conds = copy.deepcopy(conditions)
        while not_at_axis_lims:
            if axis_dir == Direction.POSITIVE:
                not_at_axis_lims = new_conds[av] + new_delta*axis_dir.value < max(self.axis_lims[av]) 
            else:
                not_at_axis_lims = new_conds[av] + new_delta*axis_dir.value > min(self.axis_lims[av])

            #If new conditions are within limits, then add the new point
            if not_at_axis_lims:
                new_conds[av] += new_delta * axis_dir.value
                _log.info(f"Force adding starting point with conditions {new_conds}")
                success = self.add_nodes_from_conditions(new_conds, None, axis_dir, True)
                if success:
                    return