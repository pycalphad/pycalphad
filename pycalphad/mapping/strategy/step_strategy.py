from typing import Union
import logging
import itertools
import copy

import numpy as np

from pycalphad import Database, variables as v
from pycalphad.core.constants import MIN_PHASE_FRACTION

from pycalphad.mapping.primitives import ZPFLine, Node, Point, ExitHint, Direction, ZPFState, _get_phase_specific_variable
import pycalphad.mapping.utils as map_utils
from pycalphad.mapping.strategy.strategy_base import MapStrategy
from pycalphad.mapping.strategy.strategy_data import SinglePhaseData, StrategyData

_log = logging.getLogger(__name__)

class StepStrategy(MapStrategy):
    def generate_automatic_starting_points(self):
        """
        Adds initial starting point in middle of free condition
        """
        av = self.axis_vars[0]
        mid_val = (self.axis_lims[av][0] + self.axis_lims[av][1]) / 2
        mid_conds = copy.deepcopy(self.conditions)
        mid_conds[av] = mid_val
        self.add_nodes_from_conditions(mid_conds, None, True)

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
        is_pot_cond = map_utils.is_state_variable(self.axis_vars[0])
        is_invariant = map_utils.degrees_of_freedom(node, self.components, self.num_potential_condition) == 0

        if num_node_cs == num_parent_cs + 1:
            # Node has more phases than parent
            if is_pot_cond and is_invariant:
                # Potential condition, create matrix for each n-1 set of phases
                node_cs_set = set(node.stable_composition_sets)
                parent_cs_set = set(node.parent.stable_composition_sets)

                # Make sure parent cs if a subset of node cs
                if len(parent_cs_set - node_cs_set) != 0:
                    return exits, exit_dirs

                # Test all n-1 set of phases excluding parent set
                for trial_stable_cs in itertools.combinations(node.stable_composition_sets, num_node_cs - 1):
                    if set(trial_stable_cs) == parent_cs_set:
                        continue
                    # comps x phases
                    phase_matrix = np.array([cs.X for cs in trial_stable_cs]).T
                    # composition list
                    global_comps = [node.get_property(v.X(e)) for e in self.elements]
                    # phase fraction
                    phase_NP = np.linalg.lstsq(phase_matrix, global_comps, rcond=None)[0].flatten()
                    if all(phase_NP > 0):
                        candidate_point = Point.with_copy(node.global_conditions, node.chemical_potentials, [], list(trial_stable_cs))
                        # Since we have the phase fraction, we can update the cs with them
                        for cs, ph_np in zip(candidate_point.stable_composition_sets, phase_NP):
                            cs.fixed = False
                            map_utils.update_cs_phase_frac(cs, ph_np)
                        exits.append(candidate_point)
                        exit_dirs.append(node.axis_direction)
                        return exits, exit_dirs

            else:
                # Not potential condition, create exit with all phases stable and free
                candidate_point = Point.with_copy(node.global_conditions, node.chemical_potentials, [], node.stable_composition_sets)
                for cs in candidate_point.stable_composition_sets:
                    cs.fixed = False
                # Add candidate point with the same direction as the node
                exits.append(candidate_point)
                exit_dirs.append(node.axis_direction)
                return exits, exit_dirs

        elif num_node_cs == num_parent_cs:
            # Number of phases are the same, remove the 0 phase
            cs_to_keep = [cs for cs in node.stable_composition_sets if cs.NP > MIN_PHASE_FRACTION]
            # If there are more than 1 zero phase, then return the empty exits, here, a new starting point should be generated
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
            # Test direction failed, so add a new starting point
            self._add_starting_point_at_last_condition(exit_point.global_conditions, proposed_direction)
            return None
        else:
            # Return axis variable, proposed direction and axis delta
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
            # ZPF line failed, so add a new starting point
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

            # If new conditions are within limits, then add the new point
            if not_at_axis_lims:
                new_conds[av] += new_delta * axis_dir.value
                _log.info(f"Force adding starting point with conditions {new_conds}")
                success = self.add_nodes_from_conditions(new_conds, axis_dir, True)
                if success:
                    return

    def get_data(self, x: v.StateVariable, y: v.StateVariable, global_x: bool = False, global_y: bool = False, set_nan_to_zero: bool = False) -> StrategyData:
        """
        Utility function to get data from StepStrategy for plotting.

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
        set_nan_to_zero : bool
            If True, NaN values will be set to zero in the data, defaults to false

        Returns
        -------
        StrategyData
            If x and y are unique for the system, then the data in StrategyData will be labeled as "SYSTEM"
            otherwise, the data in StrategyData will be labeled for all stable phases
            Examples of unique variables:

            - State variables (N, P, T)
            - Values such as GM, HM, CPM
            - Phase specific variable (i.e. v.X(phase, component) or v.NP(phase))
              This is in contrast to v.X(component) or v.NP where composition or phase fraction will be
              taken for all stable phases
        """
        if hasattr(x, 'phase_name') and x.phase_name is None:
            if not global_x:
                x = copy.deepcopy(x)
                x.phase_name = '*'

        if hasattr(y, 'phase_name') and y.phase_name is None:
            if not global_y:
                y = copy.deepcopy(y)
                y.phase_name = '*'

        # if x and y are to be computed for the entire system, the we only have a single x vs y line
        # as opposed to something like v.T vs v.NP(*) where we have a T vs NP line for each phase
        # we determine whether x or y is computed globally based of whether or not the phase_name is a wildcard
        global_x = getattr(x, 'phase_name', None) != '*'
        global_y = getattr(y, 'phase_name', None) != '*'
        if global_x and global_y:
            phases = ['SYSTEM']
        else:
            # Get all phases in strategy (including multiplicity)
            phases = sorted(self.get_all_phases())

        # For each phase, grab x and y values and plot, setting all nan values to 0 (if phase is unstable in zpf line, it will return nan for any variable)
        # Then get the max and min of x and y values to update xlim and ylim
        data = []

        for p in phases:
            x_array = []
            y_array = []
            for zpf_lines in self.zpf_lines:
                x_data = zpf_lines.get_var_list(_get_phase_specific_variable(p, x))
                y_data = zpf_lines.get_var_list(_get_phase_specific_variable(p, y))
                if set_nan_to_zero:
                    x_data[np.isnan(x_data)] = 0
                    y_data[np.isnan(y_data)] = 0
                x_array.append(x_data)
                y_array.append(y_data)

            # We return a single x, y array for all zpf_lines per phase
            x_array = np.concatenate(x_array, axis=0)
            y_array = np.concatenate(y_array, axis=0)

            # Sort arrays by x
            argsort = np.argsort(x_array)
            x_array = x_array[argsort]
            y_array = y_array[argsort]

            data.append(SinglePhaseData(p, x_array, y_array))

        return StrategyData(data)