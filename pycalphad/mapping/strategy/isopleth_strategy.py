from typing import Union
import logging
import itertools
import copy

import numpy as np

from pycalphad import Database, variables as v
from pycalphad.core.composition_set import CompositionSet

from pycalphad.mapping.primitives import ZPFLine, Node, Point, ExitHint, Direction, MIN_COMPOSITION
import pycalphad.mapping.zpf_equilibrium as zeq
import pycalphad.mapping.utils as map_utils
from pycalphad.mapping.strategy.strategy_base import MapStrategy
from pycalphad.mapping.strategy.step_strategy import StepStrategy

_log = logging.getLogger(__name__)

def _point_slope(point: Point, axis_vars: list[v.StateVariable], norm: dict[v.StateVariable, float]):
    """
    For a point with a fixed phase, get the slope d av_1 / d av_2 to determine the best axis to step along

    Unlike the binary strategy, we know what composition set to fixed since it's defined from the exit
    """
    _log.info(f"Testing point derivative {point.fixed_phases}, {point.free_phases}, {point.global_conditions}")

    axis_vars = map_utils._sort_axis_by_state_vars(axis_vars)

    av_options = [[axis_vars[0], axis_vars[1]], [axis_vars[1], axis_vars[0]]]
    options_tests = []
    for av_list in av_options:
        der = abs(zeq.compute_derivative(point, av_list[0], av_list[1]))
        der *= norm[av_list[1]] / norm[av_list[0]]
        options_tests.append((der, av_list[0]))

    best_index = -1
    best_der = np.inf
    for i in range(len(options_tests)):
        _log.info(f"Option: Axis var {options_tests[i][1]}, derivative {options_tests[i][0]}")
        if options_tests[i][0] > 1:
            if options_tests[i][0] < best_der:
                best_index = i
                best_der = options_tests[i][0]

    # Only return axis variable to step against
    if best_index == -1:
        return options_tests[0][1]
    else:
        return options_tests[best_index][1]

class IsoplethStrategy(MapStrategy):
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

                # Get all nodes that has a parent. We set axis variable to None so that the node will find a good starting direction
                # NOTE: if a stepping has a lot of failed equilibrium calculations, it's possible that the all nodes are generated
                #      as starting points (which has no parents), so no starting points for isopleth mapping would be added
                for node in step.node_queue.nodes:
                    if node.parent is not None:
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


    def _find_exits_from_node(self, node: Node):
        """
        Exits will depend on whether node is invariant or not
        For invariant, there's a max of 2*p exits (however, some of them will not be on the isopleth line and can be ignored)
        For non-invariants, a node will always have 4 exits, with one of them being the zpf line that the node was found in
        """
        exits, exit_dirs = super()._find_exits_from_node(node)
        if node.exit_hint == ExitHint.POINT_IS_EXIT:
            return exits, exit_dirs

        node_is_invariant = map_utils.degrees_of_freedom(node, self.components, self.num_potential_condition) == 0

        # Exits for invariant and non-invariant nodes are a bit long, so splitting them to individual functions
        if node_is_invariant:
            self._invariant_exits(node, exits, exit_dirs)
        else:
            self._non_invariant_exits(node, exits, exit_dirs)

        return exits, exit_dirs

    def _invariant_exits(self, node: Node, exits: list[Point], exit_dirs: list[Direction]):
        """
        Maximum number of exits is 2*p - a ZPF line entering and exiting the node for each phase
        We test the exits by going through combinations of n-1 phases (assume a and b are fixed or forbidden)
          Create a matrix of NP*x = X and test if NP is positive for all phases
          If this is true, then we create two exits, one with (n-1) (a) and (n-1) (b)
          Rather than keeping track of the two missing phases, we'll create combinations of all composition sets
              Then the nth phase will be fixed to 0 and n+1 th phase will be ignored if the candidate point passes
              This leads to some double calculations, but the linear system we solve is pretty small, so whatever
        """
        for trial_stable_compsets in itertools.permutations(node.stable_composition_sets, len(node.stable_composition_sets)):
            phase_NP = self._invariant_phase_fractions(node, trial_stable_compsets[:-2])
            if phase_NP is None:
                continue

            if all(phase_NP > 0):
                # Set fixed phase as the phase in the trail comp set that we didn't test for
                candidate_point = Point.with_copy(node.global_conditions, node.chemical_potentials, [trial_stable_compsets[-2]], list(trial_stable_compsets[:-2]))

                for cs in candidate_point._fixed_composition_sets:
                    cs.fixed = True
                    map_utils.update_cs_phase_frac(cs, 0.0)

                # Minimum phase amount set to 1e-6 (this is to account for pycalphad ignoring all phases with NP=0 when computing deltas and causing indexing issues)
                i = 0
                for cs in candidate_point._free_composition_sets:
                    cs.fixed = False
                    map_utils.update_cs_phase_frac(cs, np.amax([phase_NP[i], 1e-6]))
                    i += 1

                # Update axis variables with new point
                for av in self.axis_vars:
                    candidate_point.global_conditions[av] = candidate_point.get_property(av)

                # Check if a) candidate has already been added as an exit (does this really happen?) and b) node has already encountered this exit
                added = any(candidate_point.compare_consider_fixed_cs(candidate_exits) for candidate_exits in exits)
                exit_has_been_encountered = node.has_point_been_encountered(candidate_point, True)
                if not added and not exit_has_been_encountered:
                    _log.info(f"Found candidate exit: {candidate_point.fixed_phases}, {candidate_point.free_phases}, {candidate_point.global_conditions}")
                    exits.append(candidate_point)
                    exit_dirs.append(None)

    def _invariant_phase_fractions(self, node: Node, trial_stable_compsets: list[CompositionSet]):
        """
        Given a list of composition sets, compute the phase fraction for each phase using the fixed node global conditions

        Global conditions will account for v.N (sum of phase fractions == 1) and v.X (or linear combination of v.X)
            v.N is also included to make the matrix full rank
            This ignores the axis variables we're mapping against along with v.T and v.P

        Linear system will be composed of equations of sum(condition_alpha * NP_alpha) = condition_global

        Since the exits from an invariant is n-2 free phases + 1 fixed phase, we need C-1 fixed conditions (assuming we're mapping along one potential and one composition variable)
            C is number of components
            For now, this is essentially v.N + C-2 composition or linear combination variables

        NOTE: this is its own function since we use this for plotting as well
        """
        fixed_var = [av for av in node.global_conditions if (av != v.T and av != v.P and av not in self.axis_vars)]

        # Phase matrix is all conditions of fixed variable (for v.X, we use the composition set value, rather than global)
        # b is the value of the global condition
        phase_X_matrix = np.zeros((len(fixed_var), len(trial_stable_compsets)))
        b = np.zeros((len(fixed_var), 1))
        for i in range(len(fixed_var)):
            for j in range(len(trial_stable_compsets)):
                phase_X_matrix[i,j] = node.get_local_property(trial_stable_compsets[j], fixed_var[i])
            b[i,0] = np.squeeze(node.global_conditions[fixed_var[i]])

        # If the matrix is not full rank, then ignore it as a potential exit
        if np.linalg.matrix_rank(phase_X_matrix) != phase_X_matrix.shape[0]:
            return None

        # Exit is valid if phase fractions are positive (we don't check if phase fraction > 1 since it will sum to 1 from the v.N condition)
        phase_NP = np.matmul(np.linalg.inv(phase_X_matrix), b).flatten()
        return phase_NP

    def _non_invariant_exits(self, node: Node, exits: list[Point], exit_dirs: list[Direction]):
        """
        For non-invariant cases, the node will have two fixed phases
        There will always be three exits
        At any intersections, there are 4 regions, two regions opposite with the same phase and two regions opposite that differ by one
        For phases a and b that are fixed at the intersection and P being any number of phases, the 4 zpf lines will be
           (P, b) (a)     Crosses P+a+b -> P+b
           (P, a) (b)     Crosses P+a+b -> P+a
           (P)    (a)     Crosses P+a -> P
           (P)    (b)     Crosses P+b -> P
        """
        for i in range(2):
            # (P) (a) or (P) (b) exit
            candidate_point = Point.with_copy(node.global_conditions, node.chemical_potentials, [node.fixed_composition_sets[i]], node.free_composition_sets)
            for cs in candidate_point._free_composition_sets:
                map_utils.update_cs_phase_frac(cs, np.amax([cs.NP, 1e-6]))
            if not node.has_point_been_encountered(candidate_point, True):
                _log.info(f"Found candidate exit: {candidate_point.fixed_phases}, {candidate_point.free_phases}, {candidate_point.global_conditions}")
                exits.append(candidate_point)
                exit_dirs.append(None)

            # (P, b) (a) or (P, a) (b) exit
            candidate_point = Point.with_copy(node.global_conditions, node.chemical_potentials, [node.fixed_composition_sets[i]], node.free_composition_sets + [node.fixed_composition_sets[1-i]])
            for cs in candidate_point._free_composition_sets:
                cs.fixed = False
            for cs in candidate_point._free_composition_sets:
                    map_utils.update_cs_phase_frac(cs, np.amax([cs.NP, 1e-6]))
            if not node.has_point_been_encountered(candidate_point, True):
                _log.info(f"Found candidate exit: {candidate_point.fixed_phases}, {candidate_point.free_phases}, {candidate_point.global_conditions}")
                exits.append(candidate_point)
                exit_dirs.append(None)

    def _determine_start_direction(self, node: Node, exit_point: Point, proposed_direction: Direction):
        """
        For stepping, only one direction is possible from a node since we either step positive or negative

        If a direction cannot be found, then we force add a starting point just past the exit_point
        """
        if proposed_direction is None:
            directions = [Direction.POSITIVE, Direction.NEGATIVE]
        else:
            directions = [proposed_direction]

        # Sort exit point to fix composition set that varies the least
        norm = {av: self.normalize_factor(av) for av in self.axis_vars}
        # Axis variable is determined by dot derivative at the point (unlike binary, we know which composition set to fix based off the exit)
        axis_var = _point_slope(exit_point, self.axis_vars, norm)
        for d in directions:
            dir_results = self._test_direction(exit_point, axis_var, d)
            if dir_results is not None:
                av_delta, other_av_delta = dir_results
                _log.info(f"Found direction: {axis_var, d, av_delta} for point {exit_point.fixed_phases}, {exit_point.free_phases}, {exit_point.global_conditions}")
                return exit_point, axis_var, d, av_delta

        return None
    
    def get_zpf_data(self, x: v.StateVariable, y: v.StateVariable):
        """
        Creates dictionary of data for plotting zpf lines for isopleths

        Parameters
        ----------
        strategy : BinaryStrategy or TernaryStrategy
        x : v.StateVariable
        y : v.StateVariable

        Returns
        -------
        zpf_data : {
            "data" : [
                {
                    "phase" : str
                    "x" : [float]
                    "y" : [float]
                }
            ]
            "xlim" : [float]
            "ylim" : [float]
        }
        Phase in "data" is the fixed phase at zero-phase fraction
        """
        xlim = [np.inf, -np.inf]
        ylim = [np.inf, -np.inf]
        data = []
        for zpf_line in self.zpf_lines:
            zero_phase = zpf_line.fixed_phases[0]
            x_data = zpf_line.get_var_list(x)
            y_data = zpf_line.get_var_list(y)

            zpf_data = {
                'phase': zero_phase,
                'x': x_data,
                'y': y_data,
            }
            data.append(zpf_data)

            xlim[0] = np.amin([xlim[0], np.amin(x_data[~np.isnan(x_data)])])
            xlim[1] = np.amin([xlim[1], np.amin(x_data[~np.isnan(x_data)])])
            ylim[0] = np.amin([ylim[0], np.amin(y_data[~np.isnan(y_data)])])
            ylim[1] = np.amin([ylim[1], np.amin(y_data[~np.isnan(y_data)])])

        zpf_data = {
            'data': data,
            'xlim': xlim,
            'ylim': ylim,
        }

        return zpf_data

    def get_invariant_data(self, x: v.StateVariable, y: v.StateVariable):
        """
        Creates dictionary of data for plotting invariants for isopleths

        End points of the node is adjusted to be the intersection of the isopleth polytope on the node in n-composition space

        Parameters
        ----------
        strategy : BinaryStrategy or TernaryStrategy
        x : v.StateVariable
        y : v.StateVariable

        Returns
        -------
        node_data : [
            {
                "x" : [float]
                "y" : [float]
            }
        ]
        """
        node_data = []
        for node in self.node_queue.nodes:
            is_invariant = map_utils.degrees_of_freedom(node, self.components, self.num_potential_condition) == 0
            if is_invariant:
                x_vals = []
                y_vals = []
                for trial_stable_compsets in itertools.permutations(node.stable_composition_sets, len(node.stable_composition_sets)-2):
                    # Get phase fraction for combination of phases and node conditions
                    phase_NP = self._invariant_phase_fractions(node, trial_stable_compsets)
                    if phase_NP is None:
                        continue

                    # If phase combination is value, then extract x and y values
                    if all(phase_NP > 0):
                        if map_utils.is_state_variable(x):
                            x_vals.append(node.get_property(x))
                        else:
                            x_vals.append(sum(node.get_local_property(cs, x)*cs_NP for cs, cs_NP in zip(trial_stable_compsets, phase_NP)))
                        if map_utils.is_state_variable(y):
                            y_vals.append(node.get_property(y))
                        else:
                            y_vals.append(sum(node.get_local_property(cs, y)*cs_NP for cs, cs_NP in zip(trial_stable_compsets, phase_NP)))

                for p1, p2 in itertools.combinations(range(len(x_vals)), 2):
                    x_data = [x_vals[p1], x_vals[p2]]
                    y_data = [y_vals[p1], y_vals[p2]]
                    data = {
                        'x': x_data,
                        'y': y_data,
                    }
                    node_data.append(data)
        return node_data
