from copy import deepcopy
import itertools
from typing import List, Mapping, Tuple, Optional, Sequence
import numpy as np
from pycalphad.core.solver import Solver
from pycalphad.core.composition_set import CompositionSet
from pycalphad import calculate, variables as v

from pycalphad.mapping.primitives import Direction, ZPFLine, Point, Node, NodeQueue, NodesExhaustedError, STATEVARS, _get_global_value_for_var, _get_value_for_var
from pycalphad.mapping.custom_add_new_phases import add_new_phases
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.utils import instantiate_models, filter_phases, unpack_components


class FailedLineTermination(Exception):
    pass


class SuccessfulLineTermination(Exception):
    pass


def extract_compsets(all_compsets, result_compsets):
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


def phases_changed(initially_stable_compsets, final_stable_compsets):
    # Relies on object identity - no copies
    if set(initially_stable_compsets) != set(final_stable_compsets):
        return True
    else:
        return False


def find_node(
    desired_stable_compsets: List[CompositionSet],
    desired_metastable_compsets: List[CompositionSet],
    conditions: Mapping[v.StateVariable, float],
    axis_var: v.StateVariable,
    orig_compsets: Tuple[Sequence[CompositionSet], Sequence[CompositionSet], Sequence[CompositionSet]],  # Stable fixed, stable free, metastable
    ) -> Optional[Node]:
    solver = Solver(remove_metastable=True, allow_changing_phases=False)
    solution_compsets = [cs for cs in desired_stable_compsets]

    # set exactly one of the new combined set of phases to zero
    for cs in desired_stable_compsets:
        if not cs.fixed:
            cs.fixed = True
            cs.update(cs.dof[len(STATEVARS):], 0.0, np.asarray([conditions[sv] for sv in STATEVARS], dtype=float))
            break
    # Force at least one phase to start with nonzero amount
    if all(cs.NP==0.0 for cs in desired_stable_compsets):
        for cs in desired_stable_compsets:
            if not cs.fixed:
                cs.NP = 1.0
                break
    # correspondingly, release a condition, the current axis variable
    new_node_conds = deepcopy(conditions)
    del new_node_conds[axis_var]
    result = solver.solve(solution_compsets + desired_metastable_compsets, {str(ky): vl for ky, vl in new_node_conds.items()})
    if not result.converged:
        return None  # TODO: raise an Exception for more resolution in error handling?
    if phases_changed(desired_stable_compsets, solution_compsets):
        return None  # TODO: raise an Exception for more resolution in error handling?
    # extract the solved axis variable and put it back in the conditions
    if axis_var in STATEVARS:
        # Get the state variable from a single composition set
        new_node_conds[axis_var] = _get_value_for_var(solution_compsets[0], axis_var)
    else:
        # Get it from all composition sets, e.g. v.X() condition
        new_node_conds[axis_var] = sum(_get_value_for_var(cs, axis_var)*cs.NP for cs in solution_compsets)
    parent = Point(new_node_conds, *orig_compsets)
    new_node = Node(new_node_conds, *extract_compsets(desired_stable_compsets, solution_compsets), parent)
    return new_node


def take_step(prev_point, axis_var, axis_delta, axis_limits, direction):
    # Copy because we don't want to modify the existing point
    compsets_fixed = deepcopy(prev_point.fixed_composition_sets)
    compsets_free = deepcopy(prev_point.free_composition_sets)
    compsets_metastable = deepcopy(prev_point.metastable_composition_sets)
    conditions = deepcopy(prev_point.global_conditions)
    compsets_stable = compsets_fixed + compsets_free
    trial_compsets = compsets_stable + compsets_metastable
    if axis_var in conditions:
        # Usually this is the case, unless we changed axis variables
        conditions[axis_var] += axis_delta * direction.value
        # Bounds check conditions
        if conditions[axis_var] < axis_limits[0]:
            msg = f"Trial axis variable {axis_var} is below axis limit of {axis_limits[0]} with value {conditions[axis_var]}"
            print(f"TERMINATING LINE: {msg}")
            raise SuccessfulLineTermination(msg)
        if conditions[axis_var] > axis_limits[1]:
            msg = f"Trial axis variable {axis_var} is above axis limit of {axis_limits[1]} with value {conditions[axis_var]}"
            print(f"TERMINATING LINE: {msg}")
            raise SuccessfulLineTermination(msg)
    else:
        raise NotImplementedError("Switching axis variables is not yet supported")
    if axis_var in STATEVARS:
        # we need to update the composition set state variables
        # by definition, axis_var is a condition, so we can re-use the previously set conditions
        for cs in trial_compsets:
            # Preserve the initial DOF
            cs.update(cs.dof[len(STATEVARS):], cs.NP, np.asarray([conditions[sv] for sv in STATEVARS], dtype=float))

    # we need to make a new list, as metastable compsets will be removed, but the underlying objects should not be copied (i.e. make a shallow copy)
    solution_compsets = [x for x in trial_compsets]
    solver = Solver(remove_metastable=True, allow_changing_phases=False)
    result = solver.solve(solution_compsets, {str(ky): vl for ky, vl in conditions.items()})
    return result, solution_compsets, conditions, compsets_fixed, compsets_free, compsets_metastable


def check_global_min(stable_compsets, conditions, chemical_potentials, phase_records, temporary_global_min_helper_objs, verbose=False) -> bool:
    """
    If False, we did _not_ find global min and stable_compsets are updated accordingly  """
    grid_opts = {'pdens': 500}
    for sv in STATEVARS:
        # Use current state variables from compsets, if not specified as condition
        grid_opts[str(sv)] = conditions.get(sv, _get_value_for_var(stable_compsets[0], sv))

    grid = calculate(
        temporary_global_min_helper_objs["dbf"],
        temporary_global_min_helper_objs["comps"],
        temporary_global_min_helper_objs["phases"],
        model=temporary_global_min_helper_objs["models"],
        phase_records=phase_records,
        fake_points=False,
        output='GM',
        to_xarray=False,
        **grid_opts
        )
    added_new_phases = add_new_phases(stable_compsets, [], phase_records,
                                    grid, tuple([0] * len(STATEVARS)), chemical_potentials,
                                    np.atleast_1d([grid_opts[str(sv)] for sv in STATEVARS]),
                                    1e-4, verbose)
    is_global_min = not added_new_phases
    return is_global_min


def map_line(input_conditions, initial_point, axis_var, direction, phase_records, temporary_global_min_helper_objs):
    GLOBAL_CHECK_INTERVAL = 1
    STEP_SIZE_REDUCTION_FACTOR = 0.5
    # Terminate if current condition_delta is has been reduced below this factor
    MAX_STEP_SIZE_REDUCTION_FACTOR = 0.1  # >=1.0 means no step size reductions allowed
    condition_delta = input_conditions[axis_var][2]  # TODO: assumes conditions are given as (start, stop, step)
    smallest_allowed_step_size = MAX_STEP_SIZE_REDUCTION_FACTOR * condition_delta
    current_step_size = condition_delta

    fixed_phases = [cs.phase_record.phase_name for cs in initial_point.fixed_composition_sets]
    free_phases = [cs.phase_record.phase_name for cs in initial_point.free_composition_sets]

    zpf_line = ZPFLine(fixed_phases, free_phases)
    zpf_line.append(initial_point)
    new_node = None
    result = None
    num_steps = 0
    axis_limits = (input_conditions[axis_var][0], input_conditions[axis_var][1])
    while True:
        # TODO: assumes (start, stop, step conditions)
        try:
            step_result = take_step(zpf_line.points[-1], axis_var, current_step_size, axis_limits, direction)
        except SuccessfulLineTermination:
            # no new points to add because this can only be triggered by going out of bounds
            break
        result, solution_compsets, conditions, orig_fixed_compsets, orig_free_compsets, orig_metastable_compsets = step_result
        compsets_fixed = orig_fixed_compsets
        compsets_free = orig_free_compsets
        compsets_metastable = orig_metastable_compsets
        orig_compsets_tup = (orig_fixed_compsets, orig_free_compsets, orig_metastable_compsets)
        compsets_stable = compsets_fixed + compsets_free
        if not result.converged:
            # TODO: check if we are close to the edge of the diagram and break cleanly
            # state variables (P, T) should (probably?) be bounded by users
            # composition conditions

            # TODO: we could also be crossing some critical point (although that should be marked by a phase change to single phase?)
            # Maybe this mode can be detected by compositions converging to a tolerance?
            if current_step_size <= smallest_allowed_step_size:
                print(f"TERMINATING LINE: Convergence failure, step size for axis variable \"{axis_var}\" is below the smallest allowed (current: {current_step_size}, allowed: {smallest_allowed_step_size}): {conditions}, {compsets_stable}, {solution_compsets}")
                break
            else:
                current_step_size *= STEP_SIZE_REDUCTION_FACTOR
                continue  # the next step will use the smaller step size
        if phases_changed(compsets_stable, solution_compsets):
            # Try to find a new node
            set_compsets_stable = set(compsets_stable)
            set_solution_compsets = set(solution_compsets)
            new_stable_compsets = [cs for cs in (set_compsets_stable | set_solution_compsets)]
            desired_metastable_compsets = []  # TODO: preserves some old behavior, but we probably want to include some metastable here
            num_different_phases = len(new_stable_compsets) - len(set_compsets_stable)
            assert num_different_phases == 1, f"Expected that, at most, there's one different phase. Got {num_different_phases} different phases from input phases ({set_compsets_stable}) to output ({set_solution_compsets})."
            new_node = find_node(new_stable_compsets, desired_metastable_compsets, conditions, axis_var, orig_compsets_tup)
            if new_node is not None:
                print(f"TERMINATING LINE: Found new node {new_node}")
                break
            else:
                # TODO: Can we give a better error message?
                # TODO: do we just keep going in this case?
                # such as: assume a false positive or that on the next iteration we'll have a better chance?
                # or is it possible for a new phase to show up via global min and it _not_ be an indication of a new node?
                print(f"TERMINATING LINE: Phases changed and we were unable to find a new node.", conditions, compsets_stable, solution_compsets)
                break

        # Global min check
        verbose = False
        if num_steps % GLOBAL_CHECK_INTERVAL == 0:
            # Note that compsets_stable == solution_compsets because we passed the phases_changed check
            new_stable_compsets = [cs for cs in compsets_stable]  # shallow copy
            is_global_min = check_global_min(new_stable_compsets, conditions, result.chemical_potentials, phase_records, temporary_global_min_helper_objs, verbose=verbose)

            if not is_global_min:
                # Assume that a new phase via global grid means that there's a new node to find.
                set_compsets_stable = set(compsets_stable)
                set_new_stable_compsets = set(new_stable_compsets)
                new_stable_compsets = [cs for cs in (set_compsets_stable | set_new_stable_compsets)]
                desired_metastable_compsets = []  # TODO: preserves some old behavior, but we probably want to include some metastable here
                num_different_phases = len(new_stable_compsets) - len(set_compsets_stable)
                # TODO: will this allow trying a bunch of new phases? Is that a bad thing?
                # assert num_different_phases == 1, f"Expected that, at most, there's one different phase. Got {num_different_phases} different phases from input phases ({set_compsets_stable}) to output ({set_new_stable_compsets})."
                print(f"Metastability detected; attempting to find node. Added {num_different_phases} candidate composition sets: ({set_new_stable_compsets - set_compsets_stable}) ")
                new_node = find_node(new_stable_compsets, desired_metastable_compsets, conditions, axis_var, orig_compsets_tup)
                if new_node is not None:
                    print(f"TERMINATING LINE: Found new node {new_node}")
                    break
                else:
                    # TODO: Can we give a better error message?
                    print(f"TERMINATING LINE: Phases changed and we were unable to find a new node.", conditions, compsets_stable, new_stable_compsets)
                    break

        new_point = Point(conditions, compsets_fixed, compsets_free, compsets_metastable)
        zpf_line.append(new_point)
        num_steps += 1

        # TODO: re-organize this exit condition
        # Exit if the phase compositions are within tolerance
        # We can assume there's more then one phase because it should be impossible to
        # reach this point without flagging for a change in phases
        TOL_CLOSE = 1e-8
        if all(np.allclose(compsets_stable[0].X, cs.X, atol=TOL_CLOSE) for cs in compsets_stable[1:]):
            print(f"TERMINATING LINE: All stable composition sets have phase compositions within tolerance ({TOL_CLOSE}): {compsets_stable}")
            break

        # TODO: if there's a large change in phase composition, that may indicate crossing an invariant line with a miscibility gap, i.e. Al-Zn (FCC_A1 -> FCC_A1 + HCP_A3)

    if new_node is not None:
        # If there's a new node, it could be "behind" us in the axis variable
        # i.e. we were mapping in some metastable region before finding the new node.
        # We want to remove all Points that are past the node that we found
        # TODO: maybe we can make this logic only apply to where we found metastability?
        node_axis_value = _get_global_value_for_var(new_node, axis_var)
        # reversed so we don't change the size of the list
        for idx, point in reversed(list(enumerate(zpf_line.points))):
            point_axis_value = _get_global_value_for_var(point, axis_var)
            if direction is Direction.NEGATIVE:
                if point_axis_value < node_axis_value:
                    del zpf_line.points[idx]
                # TODO: assuming points are ordered w.r.t. axis variable, so we can stop here?
            else:  # direction == Direction.POSITIVE
                if point_axis_value > node_axis_value:
                    del zpf_line.points[idx]
                # TODO: assuming points are ordered w.r.t. axis variable, so we can stop here?
        # Add the parent Point to the ZPF line, this helps the line connectivity in plotting
        zpf_line.append(new_node.parent)

    return zpf_line, new_node


def find_exits(
    node: Node,
    stepping_mode: Optional[bool] = False,
    tielines_in_plane: Optional[bool] = False
    ) -> List[Point]:
    """Return a list of starting points and directions from a node.

    Parameters
    ----------
    tielines_in_plane : Optional[bool]
        If true, we only need to fix one phase for every pair of composition sets

    """
    node_is_invariant = False  # TODO: determine whether there's an invariant reaction based on Gibbs phase rule
    node_exits = []
    if stepping_mode:
        # 1 exit
        raise NotImplementedError("Stepping not implemented")
    elif tielines_in_plane:
        # Exactly 2 exits
        # This node must be a three phase equilibrium, where exits are pairs of phases
        assert len(node.stable_composition_sets) == 3, f"Expected exactly three phases in equilibrium if tie-lines are in the plane. Got {len(node.stable_composition_sets)} with {node.stable_composition_sets}."
        for cs_1, cs_2 in itertools.combinations(node.stable_composition_sets, 2):
            candidate_point = Point.with_copy(node.global_conditions, [cs_1], [cs_2], node.metastable_composition_sets)
            candidate_point.fixed_composition_sets[0].fixed = True
            candidate_point.fixed_composition_sets[0].NP = 0.0
            candidate_point.free_composition_sets[0].fixed = False
            candidate_point.free_composition_sets[0].NP = 1.0
            if candidate_point != node.parent:
                # this is a new exit
                node_exits.append(candidate_point)
        assert len(node_exits) == 2
    elif node_is_invariant:
        raise NotImplementedError("Invariant case not implemented")
    else:
        # 3 exits
        raise NotImplementedError("Other exits not implemented")
    return node_exits


def determine_direction(point, axis_var, axis_delta, axis_limits, phase_records, system_definition) -> Direction:
    direction = Direction.POSITIVE
    try:
        (result, solution_compsets, conditions, _, _, _) = take_step(point, axis_var, axis_delta, axis_limits, direction)
        is_global_min = check_global_min(solution_compsets, conditions, result.chemical_potentials, phase_records, system_definition, verbose=False)
        print(direction, axis_var, axis_delta, point, solution_compsets, is_global_min)
        if is_global_min:
            return direction
    except SuccessfulLineTermination:
        pass
    direction = Direction.NEGATIVE
    try:
        (result, solution_compsets, conditions, _, _, _) = take_step(point, axis_var, axis_delta, axis_limits, direction)
        is_global_min = check_global_min(solution_compsets, conditions, result.chemical_potentials, phase_records, system_definition, verbose=False)
        print(direction, axis_var, axis_delta, point, solution_compsets, is_global_min)
        if is_global_min:
            return direction
    except SuccessfulLineTermination:
        pass
    raise ValueError("Could not determine step direction")


class Mapper():
    def __init__(self, database, components, phases, conditions):
        self.conditions = deepcopy(conditions)
        self.zpf_lines: List[ZPFLine] = []
        self.node_queue = NodeQueue()

        self._current_axis_variable = v.T  # TODO: hardcoded

        self.components = components
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

    # TODO: make sure that defining the axis variable is compatible with the
    # fact that one axis variable was deleted from the Point's conditions and
    # it may not be the selected axis variable
    def add_nodes_from_point(self, point: Point, axis_var=None, search_directions=(Direction.POSITIVE, Direction.NEGATIVE)):
        # find any nodes from a starting point
        # the starting point must be in a multi-phase region
        if axis_var is None:
            axis_var = self._current_axis_variable
        for direction in search_directions:
            zpf_line, node = map_line(self.conditions, point, axis_var, direction, self.phase_records, self._system_definition)
            self.zpf_lines.append(zpf_line)
            if node is not None:
                self.node_queue.add_node(node)

    def add_node(self, node: Node):
        # add a node directly
        self.node_queue.add_node(node)

    def do_map(self):
        axis_limits = (self.conditions[self._current_axis_variable][0], self.conditions[self._current_axis_variable][1])
        while True:
            try:
                node = self.node_queue.get_next_node()
            except NodesExhaustedError:
                break
            node_exits = find_exits(node, tielines_in_plane=True)
            for start_point in node_exits:
                direction = Direction.NEGATIVE
                direction = determine_direction(start_point, self._current_axis_variable, self.conditions[self._current_axis_variable][2], axis_limits, self.phase_records, self._system_definition)
                zpf_line, new_node = map_line(self.conditions, start_point, self._current_axis_variable, direction, self.phase_records, self._system_definition)
                self.zpf_lines.append(zpf_line)
                if new_node is not None:
                    self.node_queue.add_node(new_node)
