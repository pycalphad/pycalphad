from pycalphad import Database, variables as v
from typing import List, Mapping
from pycalphad.mapping.primitives import ZPFLine, NodeQueue, STATEVARS, Node, Point, ZPFLine, _get_global_value_for_var, _get_value_for_var, Direction, ExitHint, NodesExhaustedError
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.utils import instantiate_models, filter_phases, unpack_components
from pycalphad.mapping.starting_points import starting_point_from_equilibrium
from pycalphad.mapping.utils import calculate_with_new_conditions, check_point_is_global_min, compare_cs_for_change_in_phases, create_node_from_different_points, degrees_of_freedom, check_point_is_valid
from copy import deepcopy
import numpy as np
from pycalphad.core.constants import COMP_DIFFERENCE_TOL
import copy

"""
Base strategy class that each strategy will derive from
"""

class MapStrategy:
    def __init__(self, database: Database, components : List, elements : List, phases: List, conditions: Mapping):
        self.conditions = deepcopy(conditions)
        self.stepping = False
        self.tielines = False

        # Get all axis conditions that are variable
        self.axis_vars = []
        self.axis_lims = {}
        self.axis_delta = {}
        for k,val in self.conditions.items():
            if isinstance(val, tuple):
                self.axis_vars.append(k)
                self.axis_lims[k] = (val[0], val[1])
                self.axis_delta[k] = val[-1]

        self.num_potential_conditions = 0
        for av in self.axis_vars:
            if av in STATEVARS:
                self.num_potential_conditions += 1

        self.components = components
        self.elements = elements
        self.phases = filter_phases(database, unpack_components(database, components), phases)
        models = instantiate_models(database, components, self.phases)
        phase_records = build_phase_records(database, components, self.phases, {v.N, v.T, v.P}, models)
        self.phase_records = phase_records

        # temporary use for global min purposes
        self._system_definition = {
            "dbf": database,
            "comps": components,
            "phases": self.phases,
            "models": models,
        }


        self.exits = []
        self.exitIndex = 0

        self.verbose = False

        self.zpf_lines: List[ZPFLine] = []
        self.node_queue = NodeQueue()

        self.MIN_DELTA_RATIO = 0.1
        self.DELTA_SCALE = 0.5

    def add_nodes_from_point(self, point: Point):
        raise NotImplementedError()

    def find_exits(self, node):
        raise NotImplementedError()

    def determine_start_direction(self, node: Node):
        raise NotImplementedError()

    def find_best_direction(self, node: Node, possible_directions):
        raise NotImplementedError()

    def test_swap_axis(self, zpfline: ZPFLine):
        raise NotImplementedError()

    def extract_system_definition(self):
        return self._system_definition['dbf'], self._system_definition['comps'], self._system_definition['phases'], self._system_definition['models']

    def log(self, *args):
        if self.verbose:
            print(*args)

    def normalize_factor(self, av):
        return self.axis_delta[av]

    def add_nodes_from_conditions(self, conditions):
        start_point = starting_point_from_equilibrium(self._system_definition['dbf'], self._system_definition['comps'], self._system_definition['phases'], conditions, None)
        self.add_nodes_from_point(start_point)

    def add_starting_point_with_axis(self, point, ax):
        self.node_queue.add_node(*self.convert_point_to_node(point, ax, Direction.POSITIVE, ExitHint.NO_EXITS))
        self.node_queue.add_node(*self.convert_point_to_node(point, ax, Direction.NEGATIVE, ExitHint.NO_EXITS))

    def add_starting_points_with_axes(self, points, axes):
        if axes is None:
            for p in points:
                self.node_queue.add_node(*self.convert_point_to_node(p, None, None, ExitHint.NO_EXITS))
        else:
            for p, a in zip(points, axes):
                self.node_queue.add_node(*self.convert_point_to_node(p, a, Direction.POSITIVE, ExitHint.NO_EXITS))
                self.node_queue.add_node(*self.convert_point_to_node(p, a, Direction.NEGATIVE, ExitHint.NO_EXITS))

    def convert_point_to_node(self, point: Point, start_ax, start_dir, exit_hint = ExitHint.NORMAL):
        new_point = Point(point.global_conditions, [], [cs for cs in point._free_composition_sets if not cs.fixed], point.metastable_composition_sets)
        return self._create_node(point, new_point, start_ax, start_dir, exit_hint)

    def copy_node_with_new_start_dir(self, node: Node, start_ax, start_dir, exit_hint = ExitHint.NORMAL):
        return self._create_node(node, node.parent, start_ax, start_dir, exit_hint)

    def _create_node(self, point, parent, start_ax, start_dir, exit_hint = False):
        new_node = Node(point.global_conditions, point._fixed_composition_sets, point._free_composition_sets, point.metastable_composition_sets, parent)
        new_node.axis_var = start_ax
        new_node.axis_direction = start_dir
        f = degrees_of_freedom(new_node, self._system_definition['comps'], self.num_potential_conditions)
        if exit_hint == ExitHint.NO_EXITS:
            if f != 0 or self.stepping:
                new_node.exit_hint = ExitHint.NO_EXITS
            else:
                new_node.exit_hint = ExitHint.FORCE_ALL_EXITS
        else:
            new_node.exit_hint = ExitHint.NORMAL
        return new_node, f != 0 or self.stepping

    def add_node(self, node: Node):
        """Adds node direction"""
        self.node_queue.add_node(node)

    def iterate(self):
        """
        Cases
            1. On current zpf line and it's not finished
                Check if zpf line needs to switch axis
                Attempt to take step in current set direction
                If axis limits are reached, then end zpf line
                    Else, add point to zpf line
                Update exit index counter if ZPF line finished
            2. zpf line has finished (or no zpf line exists)
                If exits remain
                    Find start direction on exit
                    If no start direction could be found, continue
                    Start zpf line from exit and direction
                Else
                    This means either no zpf lines exist, or the last zpf line has finished all exits
                    Get node from node queue
                        If no more nodes, then return True to end mapping
                    Find possible exits from node
                        Two cases:
                            Node has single defined exit (usually during stepping) - in which case there is only 1 exit
                            Node has no exit defined, which we search for exits and reset the exit index counter

        """
        # If on current zpf line
        if len(self.zpf_lines) > 0 and not self.zpf_lines[-1].finished:
            # Check direction to iterate zpf line
            self.test_swap_axis(self.zpf_lines[-1])
            av, delta, direction = self.zpf_lines[-1].axis_var, self.zpf_lines[-1].current_delta, self.zpf_lines[-1].axis_direction
            lims = self.axis_lims[av]

            # Step zpf line in current direction
            reached_axis_lims = False
            try:
                step_result = self.take_step(self.zpf_lines[-1].points[-1], av, delta, lims, direction)
            except Exception as e:
                self.log(e)
                reached_axis_lims = True

            # Check if zpf line has finished
            if reached_axis_lims:
                self.zpf_lines[-1].finished = True
            else:
                self.attempt_to_add_point(self.zpf_lines[-1], step_result)

            # If finished, then move to next exit
            if self.zpf_lines[-1].finished:
                self.log("ITERATE:\tZPF finished {}. Going to next exit".format(len(self.zpf_lines[-1].points)))
                self.exitIndex += 1

        # If no zpf lines or current zpf line has finished
        else:
            # Check if exits remain
            if self.exitIndex < len(self.exits):
                self.log("ITERATE:\tStarting zpf line from current exit {}. {}/{}".format(self.exits[self.exitIndex], self.exitIndex+1, len(self.exits)))

                # Determine a starting direction from exit
                start_dir = self.determine_start_direction(self.exits[self.exitIndex])
                if start_dir is None:
                    self.exitIndex += 1
                    return False

                # Start ZPF line
                start_node, _ = self.convert_point_to_node(self.exits[self.exitIndex], start_dir[0], start_dir[1])
                self.start_new_zpfline(start_node, start_node.start_direction(), start_dir[2])

            # No more exits
            else:
                if len(self.zpf_lines) == 0:
                    self.log("ITERATE:\tNo zpf lines, starting one")
                else:
                    self.log("ITERATE:\tNo more directions or exits. Finding exits from new node")

                # Get node from queue
                try:
                    node = self.node_queue.get_next_node()
                except NodesExhaustedError:
                    self.log("ITERATE:\tNo more nodes, ending")
                    return True

                # Find exits from new node
                self.log("ITERATE:\tAttemping to start zpf line with: ", node)
                try:
                    assert node.exit_hint != ExitHint.NO_EXITS

                    # If we find exits, then reset exit index counter and continue to next iteration
                    self.exits = self.find_exits(node)
                    self.exitIndex = 0
                    self.log("ITERATE:\t{} exits found for node".format(len(self.exits)))

                # If no exits, then check if node has a pre-defined exit and start zpf line from there
                except Exception as e:
                    if node.exit_hint == ExitHint.NO_EXITS:
                        if node.start_direction()[0] is None:
                            self.log("ITERATE:\tForce single exit, finding best direction")
                            start_dir = self.determine_start_direction(node)
                            if start_dir is None:
                                self.log("ITERATE:\tNo direction could be found")
                                return False
                            node.axis_var = start_dir[0]
                            node.axis_direction = start_dir[1]
                            self.start_new_zpfline(node, node.start_direction(), start_dir[2])
                        else:
                            self.log("ITERATE:\tUsing predefined exit {}".format(node.start_direction()))
                            starting_delta = self.axis_delta[node.axis_var] if self.stepping else self.axis_delta[node.axis_var]*self.MIN_DELTA_RATIO
                            self.start_new_zpfline(node, node.start_direction(), starting_delta)
                    else:
                        self.log(e)
                        self.log("ITERATE:\tNo exits found")


        return False

    def do_map(self):
        finished = False
        while not finished:
            finished = self.iterate()

    def start_new_zpfline(self, node, newdir, starting_delta):
        fixed_phases = [cs.phase_record.phase_name for cs in node.fixed_composition_sets]
        free_phases = [cs.phase_record.phase_name for cs in node.free_composition_sets]
        self.zpf_lines.append(ZPFLine(fixed_phases, free_phases))
        self.zpf_lines[-1].points.append(node)
        self.zpf_lines[-1].axis_var = newdir[0]
        self.zpf_lines[-1].axis_direction = newdir[1]
        self.zpf_lines[-1].current_delta = starting_delta

    def attempt_to_add_point(self, zpfline: ZPFLine, step_result):
        """
        Performs a few checks on new calculation to see if its worthy to join the zpf line
        Otherwise, we get a new node and create a new zpf line from it

        Checks include - if results converged, if number of phases stay the same and if it's still global minimum

        Post check after adding point if is phase compositions converged to the same composition
        """
        result, new_point, orig_cs = step_result

        zpfline.last_globally_checked_index += 1
        results_converged = self.check_results_converged(zpfline, step_result, self.axis_delta[zpfline.axis_var])
        if not results_converged:
            return

        num_phases_same, new_node = self.check_change_in_phases(step_result)
        if not num_phases_same and new_node is not None:
            self.process_new_node(zpfline, new_node, self.stepping)
            zpfline.finished = True
            return

        composition_is_nearby = self.check_composition(zpfline, new_point, zpfline.points[-1])
        if not composition_is_nearby:
            zpfline.finished = True
            return

        still_global_min, new_node = self.check_is_still_global_min(zpfline, step_result)
        if not still_global_min:
            if new_node is not None:
                self.process_new_node(zpfline, new_node, self.stepping)
            zpfline.finished = True
            return

        is_within_axis_limits = self.check_if_within_axis_limits(zpfline, step_result)
        if not is_within_axis_limits:
            zpfline.finished = True
            return

        phase_apart = self.check_similar_phase_compositions(new_point)
        if not phase_apart:
            self.log("TERMINATING LINE: All stable composition sets have phase compositions within tolerance")
            zpfline.finished = True
            return

        zpfline.append(new_point)

    def process_new_node(self, zpfline: ZPFLine, new_node: Node, stepping):
        """
        New node could be behind the last point in the zpf line
        However, since the axis variable could change, we'll have to compare how the direction changes between new node->last point, and last->second to last point
        """
        self.log("PROCESS:\tRemoving back nodes", len(zpfline.points))
        node_pos = []
        for av in self.axis_vars:
            if stepping:
                node_pos.append(new_node.global_conditions[av])
            else:
                node_pos.append(_get_global_value_for_var(new_node, av))
        # Reverse order, but stop at 1, since we'll run into an issue trying to find the second to last point
        for i in range(len(zpfline.points)-1, 0, -1):
            p1_pos = []
            p2_pos = []
            for av in self.axis_vars:
                if stepping:
                    p1_pos.append(zpfline.points[i].global_conditions[av])
                    p2_pos.append(zpfline.points[i-1].global_conditions[av])
                else:
                    p1_pos.append(_get_global_value_for_var(zpfline.points[i], av))
                    p2_pos.append(_get_global_value_for_var(zpfline.points[i-1], av))
            v21 = [p1_pos[j] - p2_pos[j] for j in range(len(self.axis_vars))]
            v1new = [node_pos[j] - p1_pos[j] for j in range(len(self.axis_vars))]
            # Dot product of v21 and v1new
            dp = np.dot(v21, v1new)
            # Delete points until dot product is no longer 0
            if dp < 0:
                del zpfline.points[i]
            else:
                break

        zpfline.append(new_node.parent)

        if self.stepping:
            new_node.axis_direction = zpfline.points[0].axis_direction

        if self.node_queue.add_node(new_node, stepping):
            self.log("PROCESS:\tadding node", new_node)
        else:
            self.log("PROCESS:\tNode already added", new_node)

    def test_direction(self, node, axis_var, direction, MIN_DELTA_RATIO = 1):
        av = axis_var
        d = direction
        ax_delta = self.axis_delta[av]
        while ax_delta >= self.axis_delta[av]*MIN_DELTA_RATIO:
            self.log("DIRECTION:\tAttemping step in ", av, d, ax_delta)
            viable_direction = True
            try:
                (result, new_point, orig_cs) = self.take_step(node, av, ax_delta, self.axis_lims[av], d)
                num_different_compsets = len(set(new_point.stable_composition_sets).symmetric_difference(orig_cs))
                is_global_min, new_point = self.check_global_min(new_point, result.chemical_potentials)
                if not is_global_min:
                    self.log("DIRECTION:\tNot global minimum")
                    viable_direction = False
            except Exception as e:
                self.log("DIRECTION:\t", e)
                num_different_compsets = 0
                viable_direction = False
            if num_different_compsets != 0:
                self.log("DIRECTION:\tLocal equilibrium reduced number of phases")
                viable_direction = False

            if viable_direction:
                self.log("DIRECTION:\tPossible direction found ", (av, d, ax_delta))
                return True, (av, d, ax_delta, new_point, node)
            else:
                ax_delta *= self.DELTA_SCALE
        return False, None

    def _get_exit_info_from_node(self, node):
        """
        Exit information for the node
        Includes:
            Number of components
            Number of stable phases
            Whether node is invariant (DOF = 0)
        """
        num_comps = np.asarray(node.stable_composition_sets[0].X).size
        num_stable_phases = len(node.stable_composition_sets)
        node_is_invariant = degrees_of_freedom(node, self.elements, self.num_potential_conditions) == 0
        return num_comps, num_stable_phases, node_is_invariant

    def step_condition(self, point, axis_var, axis_delta, axis_limits, direction):
        """
        Create new conditions by stepping along axis variable

        Checks will include bound constraints and composition summation constraints
        """
        new_conds = {v:k for v,k in point.global_conditions.items()}
        new_conds[axis_var] += axis_delta * direction.value

        # Check bounds - for composition, add a small offset to avoid numerical errors
        offset = 0 if axis_var in STATEVARS else 1e-6
        if new_conds[axis_var] <= axis_limits[0]:
            if new_conds[axis_var] <= axis_limits[0] - 0.9*axis_delta:
                raise Exception("TAKE_STEP:\tCondition below axis limits")
            else:
                new_conds[axis_var] = axis_limits[0] + offset
        if new_conds[axis_var] >= axis_limits[1]:
            if new_conds[axis_var] >= axis_limits[1] + 0.9*axis_delta:
                raise Exception("TAKE_STEP:\tCondition above axis limits")
            else:
                new_conds[axis_var] = axis_limits[1] - offset

        # Also check if new conditions will step out of composition constraints
        # This will be due to mapping along a composition axis when there's more than 2 components
        # This isn't the best way to do it if axis variables other than composition is present, but will do for now
        comp_sum = sum([new_conds[v] for v in new_conds if v not in STATEVARS])
        if comp_sum >= 1:
            if comp_sum >= 1 + 0.9*axis_delta:
                raise Exception("TAKE_STEP:\tComposition summation violation")
            else:
                new_conds[axis_var] = 1 - (comp_sum - new_conds[axis_var]) - offset
        return new_conds

    def take_step(self, prev_point, axis_var, axis_delta, axis_limits, direction):
        new_conds = self.step_condition(prev_point, axis_var, axis_delta, axis_limits, direction)
        free_av = None
        if not self.stepping:
            free_av = self.axis_vars[int(1-self.axis_vars.index(axis_var))]
        result, new_point, orig_cs = calculate_with_new_conditions(prev_point, new_conds, free_av)
        return result, new_point, orig_cs

    def extract_compsets(self, all_compsets, result_compsets):
        """
        all_compsets will contain both stable and metastable CompositionSet
        results_compsets only contains stable CompositionSet
        stable CompositionSets can be free or fixed
        """
        # Relies on object identity - no copies
        stable_fixed_cs = []
        stable_free_cs = []
        metastable_cs = []
        for cs in all_compsets:
            if cs in result_compsets:
                if cs.fixed:
                    stable_fixed_cs.append(cs)
                else:
                    stable_free_cs.append(cs)
            else:
                metastable_cs.append(cs)
        return stable_fixed_cs, stable_free_cs, metastable_cs

    def check_global_min(self, point, chem_pot):
        return check_point_is_global_min(point, chem_pot, self._system_definition, self.phase_records)

    def check_results_converged(self, zpfline, step_result, axis_delta):
        result, new_point, orig_cs = step_result
        if not result.converged:
            # TODO: we could also be crossing some critical point (although that should be marked by a phase change to single phase?)
            # Maybe this mode can be detected by compositions converging to a tolerance?
            if zpfline.current_delta <= self.MIN_DELTA_RATIO*axis_delta:
                self.log('CONVERGE:\tMinimim step sized reached')
                zpfline.finished = True
            else:
                self.log('CONVERGE:\tConvergence failure, reducing step size {}'.format(zpfline.current_delta))
                zpfline.current_delta *= 0.5
            return False
        else:
            return True

    def check_change_in_phases(self, step_result):
        results, new_point, orig_cs = step_result
        num_different_phases = compare_cs_for_change_in_phases(orig_cs, new_point.stable_composition_sets)

        if num_different_phases == 1:
            self.log("PH_CHANGE:\tfound different number of phases")
            assert num_different_phases == 1, f"Expected that, at most, there's one different phase"
            orig_point = Point(new_point.global_conditions, [cs for cs in orig_cs if cs.fixed], [cs for cs in orig_cs if not cs.fixed], [])
            new_node = create_node_from_different_points(orig_point, new_point, self.axis_vars, self.axis_lims)
            if new_node is not None:
                self.log("PH_CHANGE:\tNew node found", new_node)
                return False, new_node
            else:
                self.log("PH_CHANGE:\tError: new node could not be found")
                return False, None
        return True, None

    def check_composition(self, zpfline: ZPFLine, new_point, prev_point):
        """
        Checks if step result has strayed too far from the previous point, since we're limited to the axis_delta, we could use that for checking
            We'll add a scaling factor to the axis_delta for safety

        Also check if both axis are within limits
            For ternaries, since we only step along a single axis variable at a time, the second axis is free to move where it needs

        For binary phase diagrams, this should always pass
        But for ternaries, I've seen this fail under two circumstances:
            The fixed composition of one axis can correspond to two two-phase regions of the same two phases
                This sometimes happens when we step along one axis and the other axis should go past the axis limits, but instead it goes to the other two-phase region
                Ex. BCC (Fe=0.49, Co=0.01) + SIGMA (Fe=0.53, Co=0.2) -> BCC (Fe=0.50, Co=0.5) + SIGMA (Fe=49, Co=0.3)
                    Where SIGMA is fixed at 0
            We encounter an invariant with a miscibility gap, but we find this node from a two-phase region that doesn't have the miscibility gap
                Ex. FCC+BCC -> FCC+FCC+BCC
        """
        # Get axis variables of all composition sets in step result and previous point
        # Since we check if the number of phases changed first, we could use the new point composition sets rather than the original
        #    This prevents the possibility of a phase becoming unstable and having a weird composition that could cause this check to fail
        new_point_comps = {av: _get_global_value_for_var(new_point, av) for av in self.axis_vars}
        prev_point_comps = {av: _get_global_value_for_var(prev_point, av) for av in self.axis_vars}

        dist = np.amax([np.abs((new_point_comps[av] - prev_point_comps[av])/self.normalize_factor(av)) for av in self.axis_vars])

        # Check new_point_comps is within numerical limits
        # Axis limits of state variables is always 0
        # Axis limits of composition will be 0 if we're stepping along temperature, else 1e-6
        #    If we're stepping along composition, then 1e-6 is a decent place to stop
        #    If we're stepping along temperature, then we don't care about the composition as long as within the composition constraints [0,1]
        in_limit = True
        for av in self.axis_vars:
            statevar_offset = 0
            x_offset = 0 if zpfline.axis_var in STATEVARS else 1e-6
            offset = statevar_offset if av in STATEVARS else x_offset
            if new_point_comps[av] < self.axis_lims[av][0] + offset or new_point_comps[av] > self.axis_lims[av][1] - offset:
                in_limit = False
                break

        if dist > 3 or not in_limit:
            if dist > 3:
                self.log('CHECK_COMP:\tNew composition violates max stepping distance')
            else:
                self.log('CHECK_COMP:\tNew composition violates axis limits')
            self.log('CHECK_COMP:\t', new_point.free_composition_sets, prev_point.free_composition_sets)
            return False
        else:
            return True

    def check_is_still_global_min(self, zpfline, step_result):
        GLOBAL_CHECK_INTERVAL = 1
        tolerance = 1e-4
        pdens = 500
        results, new_point, orig_cs = step_result

        #Only perform the global min check every n intervals (ideally n would be > 1 for performance, but n = 1 improves stability)
        if zpfline.last_globally_checked_index % GLOBAL_CHECK_INTERVAL == 0:
            is_global_min, global_test_point = check_point_is_global_min(new_point, results.chemical_potentials, self._system_definition, self.phase_records, tolerance, pdens)

            #If new phase was detected
            if not is_global_min:
                self.log('GLOBAL_MIN:\tMetastable condition detected', global_test_point)

                #Create a deep copy of the previous point since create_node_from_different_points only creates a shallow copy
                #   This will prevent the create_node_from_different_points function from changing the composition sets in point_ref
                point_ref = copy.deepcopy(new_point)
                new_node = create_node_from_different_points(new_point, global_test_point, self.axis_vars, self.axis_lims)
                if new_node is not None:
                    #Compare the conditions on the new node and reference point, sometimes, equilibrium will leads to an unusually off set
                    #   of compositions, in this case, we return that it is not global min, but don't supply the next node. This will end
                    #   the current zpf line without an node to get exits from
                    if not self.check_composition(zpfline, new_node, point_ref):
                        self.log('GLOBAL_MIN:\tNew node was not calculated correctly', new_node)
                        return False, None

                    #If global min and the new node is considered valid, we want to end the zpf line with the new node and
                    #   get exits from it
                    self.log('GLOBAL_MIN:\tNew node from global equilibrium found', new_node)
                    return False, new_node
                else:
                    #If equilibrium did not converge, we want to end the zpf line without the new node
                    # TODO: Can we give a better error message?
                    self.log('GLOBAL_MIN:\tGlobal equilibrium found, but new node could not be found')
                    return False, None

        #If all checks pass, then current point is still global min and there's no new node to supply
        return True, None

    def check_if_within_axis_limits(self, zpfline, step_result):
        result, new_point, orig_cs = step_result
        for av in self.axis_vars:
            if av in STATEVARS:
                val = _get_value_for_var(new_point.stable_composition_sets[0], av)
            else:
                val = sum(_get_value_for_var(cs, av)*cs.NP for cs in new_point.stable_composition_sets)
            if val > self.axis_lims[av][1] or val < self.axis_lims[av][0]:
                return False
        return True

    def check_similar_phase_compositions(self, point, tol = 1e-8):
        compsets_stable = point.stable_composition_sets
        TOL_CLOSE = tol
        if len(compsets_stable) > 1:
            # If any two composition sets are near each other, we should end the zpf line since it means we're transitioning to a n-1 phase region
            if any(np.allclose(compsets_stable[i].X, compsets_stable[j].X, atol=TOL_CLOSE) for i in range(len(compsets_stable)) for j in range(i+1, len(compsets_stable))):
                return False

        return True