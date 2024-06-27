from typing import Union
import logging
import copy

import numpy as np

from pycalphad import Database, variables as v
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.utils import instantiate_models, unpack_components, filter_phases
from pycalphad.core.composition_set import CompositionSet


from pycalphad.mapping.primitives import STATEVARS, ZPFLine, NodeQueue, Node, Point, ExitHint, Direction, MIN_COMPOSITION, ZPFState
from pycalphad.mapping.starting_points import point_from_equilibrium
import pycalphad.mapping.zpf_equilibrium as zeq
import pycalphad.mapping.zpf_checks as zchk
import pycalphad.mapping.utils as map_utils

_log = logging.getLogger(__name__)

class MapStrategy:
    """
    Base strategy class for phase diagram construction

    Derived classes:
        SteppingStrategy - for single axis diagrams
        TielinesStrategy - for binary and ternary phase diagrams
                           this may be split to handle since ternary phase diagrams may need some special handling
        IsoplethStrategy - for isopleths
        
        Eventually, we will want a general strategy that can map out more than two axis variables

    
    """
    #Some default constants
    MIN_DELTA_RATIO = 0.1
    DELTA_SCALE = 0.5
    GLOBAL_CHECK_INTERVAL = 1
    GLOBAL_MIN_PDENS = 500
    GLOBAL_MIN_TOL = 1e-4
    GLOBAL_MIN_NUM_CANDIDATES = 1

    def __init__(self, dbf: Database, components: list[str], phases: list[str], conditions: dict[v.StateVariable, Union[float, tuple[float]]]):
        if isinstance(dbf, str):
            dbf = Database(dbf)
        self.dbf = dbf
        self.components = components
        self.elements = map_utils.elements_from_components(self.components)
        self.phases = filter_phases(self.dbf, unpack_components(self.dbf, self.components), phases)
        self.conditions = copy.deepcopy(conditions)

        #Add v.N to conditions. Mapping assumes that v.N is in conditions
        if v.N not in self.conditions:
            self.conditions[v.N] = 1

        self.axis_vars = []
        self.axis_lims = {}
        self.axis_delta = {}
        for key, val in self.conditions.items():
            if isinstance(val, tuple):
                self.axis_vars.append(key)
                self.axis_lims[key] = (val[0], val[1])
                self.axis_delta[key] = val[-1]

        self.num_potential_condition = sum([1 if av in STATEVARS else 0 for av in self.axis_vars])
        self.models = instantiate_models(self.dbf, self.components, self.phases)
        self.phase_records = build_phase_records(self.dbf, self.components, self.phases, set(STATEVARS), self.models)

        #In case we need to call pycalphad functions outside this class
        self.system_info = {
            "dbf": self.dbf,
            "comps": self.components,
            "phases": self.phases,
            "models": self.models,
            "phase_records": self.phase_records
        }

        self.zpf_lines: list[ZPFLine] = []
        self.node_queue = NodeQueue()

        self._current_node = None
        self._exits = []
        self._exit_dirs = []
        self._exit_index = 0

    def get_all_phases(self):
        """
        Goes through ZPF lines to get all unique phases. For miscibility gaps, phases will have #n added to it

        In some cases, there might be no ZPF lines (e.g. ternaries with all line compounds), in which case, we return an empty set
        There should always be nodes in the node_queue since it includes starting points (even if they're not nodes in the mapping sense)
        """
        if len(self.zpf_lines) > 0:
            zpf_phases = set.union(*[set(zpf_line.stable_phases_with_multiplicity) for zpf_line in self.zpf_lines])
        else:
            zpf_phases = set()
        if len(self.node_queue.nodes) > 0:
            node_phases = set.union(*[set(node.stable_phases_with_multiplicity) for node in self.node_queue.nodes])
        else:
            node_phases = set()

        return list(set.union(zpf_phases, node_phases))

    def normalize_factor(self, av):
        """
        Since potential and composition variables are on different scales, we normalize by the delta value
        """
        return self.axis_delta[av]
    
    def _other_av(self, av):
        if len(self.axis_vars) == 1:
            return None
        else:
            return self.axis_vars[1-self.axis_vars.index(av)]

    def add_nodes_from_conditions(self, conditions: dict[v.StateVariable, float], axis_var: v.StateVariable = None, direction: Direction = None, force_add: bool = True):
        """
        Computes equilibrium and creates a point from input conditions

        If a direction is supplied, then we can just add the new point with the direction
        If not, then we add two points, for each direction

        Since this is a starting point, we set the ExitHint to POINT_IS_EXIT since we assume the starting point is not a true node

        Also by default, we force add the node to skip checking if the node is already in the node queue
        """
        point = point_from_equilibrium(self.dbf, self.components, self.phases, conditions, axis_var, models=self.models, phase_record_factory=self.phase_records)
        if point is None:
            _log.warning(f"Point could not be found from {conditions}")
            return False
        if direction is None:
            _log.info(f"No direction is given, adding point from {conditions} with both directions")
            self.node_queue.add_node(self._create_node_from_point(point, None, axis_var, Direction.POSITIVE, ExitHint.POINT_IS_EXIT), force_add)
            self.node_queue.add_node(self._create_node_from_point(point, None, axis_var, Direction.NEGATIVE, ExitHint.POINT_IS_EXIT), force_add)
        else:
            self.node_queue.add_node(self._create_node_from_point(point, None, axis_var, direction, ExitHint.POINT_IS_EXIT), force_add)
        return True

    def _create_node_from_point(self, point: Point, parent: Point, start_ax: v.StateVariable, start_dir: Direction, exit_hint: ExitHint = ExitHint.NORMAL):
        """
        Given a point and a parent, create a node with a starting axis/direction

        If no parent is given, then a copy of the point without the fixed composition sets is made as the parent
        """
        #if parent is None:
        #    _log.info(f"No parent is given. Creating parent from point with {point.free_phases}")
        #    parent = Point(point.global_conditions, point.chemical_potentials, [], point.free_composition_sets)
        
        new_node = Node(point.global_conditions, point.chemical_potentials, point.fixed_composition_sets, point.free_composition_sets, parent)
        new_node.axis_var = start_ax
        new_node.axis_direction = start_dir
        new_node.exit_hint = exit_hint
        return new_node
    
    def iterate(self):
        """
        Hierarchy of mapping iterations
            1. Node queue
                If not empty, then create exits from next node
            2. Node exits
                If exits remain, then create new zpf line from next exit
            3. ZPF line
                If not finished, continue ZPF line
        """
        if len(self.zpf_lines) > 0 and self.zpf_lines[-1].status == ZPFState.NOT_FINISHED:
            self._continue_zpf_line()
        elif len(self._exits) > self._exit_index:
            _log.info("No zpf line or last zpf line has finished. Attempting to start zpf line from next exit")
            #This will start a new zpf line that is not finished
            self._start_zpf_line()
            self._exit_index += 1
        elif self.node_queue.size() > 0:
            _log.info("No more exits from current node. Attempting to get next node")
            #This will reset the _exits list and _exitIndex to 0
            self._find_node_exits()
        else:
            # Mapping is finished once the last zpf line is finished, 
            # the last node has no more exits and the node queue has no more nodes
            _log.info("No more nodes in queue. Ending mapping.")
            return True
        return False
    
    def do_map(self):
        """
        Wrapper over iterate to run until finished
        """
        finished = False
        while not finished:
            finished = self.iterate()

    def _continue_zpf_line(self):
        """
        1. If more than two free axis, check if axis needs to be swapped
              Ideally, we want the axis that will result in minimal change in the other axes
        2. Take step along current axis and direction
        3. Check if the new point generated from step can be added to the zpf line
              _attempt_to_add_point will account for invalid points and detecting new nodes
        """
        #Test if axis needs to be swapped
        self._test_swap_axis(self.zpf_lines[-1])

        #Take step
        step_results = self._step_zpf_line(self.zpf_lines[-1])

        #Check new point and try to add it to zpf line
        self._attempt_to_add_point(self.zpf_lines[-1], step_results)

    def _test_swap_axis(self, zpf_line: ZPFLine):
        """
        By default, we won"t swap axis. This will be the case for stepping
        For more than 2 axis, we do a comparison of how much each axis variable changed in the last two steps
        """
        return

    def _step_conditions(self, point: Point, axis_var: v.StateVariable, axis_delta: float, axis_lims: tuple[float], direction: Direction):        
        """
        Creates a copy of condition and steps in proposed direction

        If the axis variable is outside the defined limits, then we adjust the conditions to be at the edge of the axis limits
        """
        new_conds = copy.deepcopy(point.global_conditions)
        new_conds[axis_var] += axis_delta*direction.value
        
        #Limit axis variable to axis limits (provide a small offset for composition)
        #TODO: check if composition sums to more than 1, could be an override for isopleth strategy
        hit_axis_limit = False
        offset = 0 if axis_var in STATEVARS else MIN_COMPOSITION
        if new_conds[axis_var] > max(axis_lims) -  offset:
            new_conds[axis_var] = max(axis_lims) - offset
            hit_axis_limit = True
        if new_conds[axis_var] < min(axis_lims) + offset:
            new_conds[axis_var] = min(axis_lims) + offset
            hit_axis_limit = True
        return new_conds, hit_axis_limit

    def _take_step(self, point: Point, axis_var: v.StateVariable, axis_delta: float, axis_lims: tuple[float], direction: Direction):
        """
        Adjust conditions and compute equilibrium, this will return a new point (or None if equilibrium fails)
        """
        #Here we don't care if stepping reached the axis limit since we check this afterwards
        new_conds, hit_axis_limit = self._step_conditions(point, axis_var, axis_delta, axis_lims, direction)
        if hit_axis_limit:
            return None
        _log.info(f"Stepping point {point.fixed_phases}, {point.free_phases}, {point.global_conditions} along {axis_var}, {axis_delta*direction.value}")
        return zeq.update_equilibrium_with_new_conditions(point, new_conds, self._other_av(axis_var))

    def _step_zpf_line(self, zpf_line: ZPFLine):
        """
        This is the same as _take_step except we use the conditions from the current zpf line
        """
        return self._take_step(zpf_line.points[-1], zpf_line.axis_var, zpf_line.current_delta, self.axis_lims[zpf_line.axis_var], zpf_line.axis_direction)

    def _attempt_to_add_point(self, zpf_line: ZPFLine, step_results: tuple[Point, list[CompositionSet]]):
        """
        Go through list of check functions to see if the new point can be added
        Rules
         1. All check functions will take in the current zpf line and a tuple(new_point, [orig_cs])
         2. If the check function returns True, it means the check has passed and we move onto the next check
         3. If the check function returns False, we stop checking and end this function
               a. Fails because equilibrium did not converge or invalid condition -> adjust axis delta and try again
               b. Fails because we found a node -> backtrack zpf line to remove offending points and add node parent to zpf line
         4. If all check functions pass, then we add the point

        Not a fan of how this is implemented, but I want the API for each check function to be the same, with extra args having default values if not supplied
        """
        check_functions = [zchk.check_valid_point, zchk.check_change_in_phases, zchk.check_global_min, zchk.check_axis_values, zchk.check_similar_phase_composition]
        axis_data = {
            "axis_vars": self.axis_vars,
            "axis_delta": self.axis_delta,
            "axis_lims": self.axis_lims
        }
        extra_args = {
            "delta_scale": self.DELTA_SCALE,
            "min_delta_ratio": self.MIN_DELTA_RATIO,
            "global_check_interval": self.GLOBAL_CHECK_INTERVAL,
            "global_num_candidates": self.GLOBAL_MIN_NUM_CANDIDATES,
            "normalize_factor": {av: self.normalize_factor(av) for av in self.axis_vars},
            "system_info": self.system_info,
            "pdens": self.GLOBAL_MIN_PDENS,
            "tol": self.GLOBAL_MIN_TOL
        }
        for check in check_functions:
            new_node = check(zpf_line, step_results, axis_data, **extra_args)

            #If we found a new node, then process it and end zpf line
            if zpf_line.status == ZPFState.NEW_NODE_FOUND:
                self._process_new_node(zpf_line, new_node)
                return
            #If equilibrium failed, then reset zpf line so it can step from previous point
            elif zpf_line.status == ZPFState.ATTEMPT_NEW_STEP:
                zpf_line.status = ZPFState.NOT_FINISHED
                return
            #If anything occurs that interrupts zpf line, break out of the loop
            #Whether we add or don't add point will be decided later
            elif zpf_line.status != ZPFState.NOT_FINISHED:
                _log.info(f"ZPF line {zpf_line.fixed_phases}, {zpf_line.free_phases} ended with {zpf_line.status}")
                break

        # If the zpf line reached a limit (axis limits or similar phase composition), then we
        # can still add the point
        valid_statuses = [ZPFState.NOT_FINISHED, ZPFState.REACHED_LIMIT] 
        if zpf_line.status in valid_statuses:
            new_point, _ = step_results
            zpf_line.append(new_point)

            #If we successfully added a new point, attempt to increase the axis delta if is smaller than the default
            if zpf_line.current_delta < self.axis_delta[zpf_line.axis_var]:
                zpf_line.current_delta = np.amin([self.axis_delta[zpf_line.axis_var], zpf_line.current_delta / self.DELTA_SCALE])

    def _process_new_node(self, zpf_line: ZPFLine, new_node: Node):
        """
        Back tracks zpf line until the angle between the last edge on the zpf line and the edge connect to the node is greater than 90 (dot product > 0)

        TODO: while this should allow for performing global min check every couple iterations, this is very iffy
        """
        _log.info("Back tracking zpf line to add node")
        node_pos = np.array([new_node.get_property(av) for av in self.axis_vars])
        orig_len = len(zpf_line.points)
        for i in range(len(zpf_line.points)-1, 0, -1):
            p1 = zpf_line.points[i]
            p2 = zpf_line.points[i-1]
            p1_pos = np.array([p1.get_property(av) for av in self.axis_vars])
            p2_pos = np.array([p2.get_property(av) for av in self.axis_vars])
            v21 = p1_pos - p2_pos
            vnode1 = node_pos - p1_pos
            if np.dot(v21, vnode1) < 0:
                del zpf_line.points[i]
            else:
                break

        final_len = len(zpf_line.points)
        _log.info(f"Removed zpf points from {orig_len} to {final_len}")

        #zpf_line.append(new_node.parent)
        zpf_line.append(new_node)

        #Set axis variable and direction from the previous zpf line
        #So we know where the node came from (this can give some hints for exit finding)
        new_node.axis_var = zpf_line.axis_var
        new_node.axis_direction = zpf_line.axis_direction

        #Add to node queue
        #For stepping, we will force add. Most nodes will be unique, however, if we're stepping along composition
        #in a binary, then the nodes won't be unique
        if len(self.axis_vars) == 1:
            self.node_queue.add_node(new_node, True)
        else:
            if not self.node_queue.add_node(new_node):
                _log.info(f"Node {new_node.fixed_phases}, {new_node.free_phases} has already been added")

    def _start_zpf_line(self):
        """
        1. Gets current exit from the node
        2. Find start direction
        3. Create new zpf line with exit and direction information
        """
        #Find start direction from current exit
        direction_data = self._determine_start_direction(self._current_node, self._exits[self._exit_index], self._exit_dirs[self._exit_index])

        #If no direction can be found, then move to the next exit
        if direction_data is None:
            return
        
        #Create node from exit and start direction
        exit_point, start_ax, start_dir, start_delta = direction_data
        #start_node = self._create_node_from_point(self._exits[self._exit_index], None, start_ax, start_dir)

        #Initialize zpf line
        #self.zpf_lines.append(ZPFLine(start_node.fixed_phases, start_node.free_phases))
        #self.zpf_lines[-1].points.append(start_node)
        self.zpf_lines.append(ZPFLine(exit_point.fixed_phases, exit_point.free_phases))
        self.zpf_lines[-1].points.append(exit_point)
        self.zpf_lines[-1].axis_var = start_ax
        self.zpf_lines[-1].axis_direction = start_dir
        self.zpf_lines[-1].current_delta = start_delta

        _log.info(f"Starting zpf line with {self.zpf_lines[-1].points[-1].global_conditions}, {self.zpf_lines[-1].points[-1].fixed_phases}, {self.zpf_lines[-1].points[-1].free_phases} and {self.zpf_lines[-1].axis_var}, {self.zpf_lines[-1].current_delta*self.zpf_lines[-1].axis_direction.value}")
        _log.info(f"ZPF starting from node {self._current_node.fixed_phases}, {self._current_node.free_phases}, {self._current_node.global_conditions}")

    def _determine_start_direction(self, node: Node, exit_point: Point, proposed_direction: Direction = None):
        """
        From an exit point, this will return an axis variable, direction and delta
        that is suggested to give a good starting direction to a ZPF line

        If proposed_direction is None, this will test both positive and negative, 
        otherwise, it will only test the proposed direction
        """
        raise NotImplementedError()

    def _find_node_exits(self):
        """
        1. Get next node in queue
        2. Find exits from node (implemented for each strategy)
        """
        #Get new node from node queue
        self._current_node = self.node_queue.get_next_node()
        _log.info(f"Finding exits from node {self._current_node.fixed_phases}, {self._current_node.free_phases}, {self._current_node.global_conditions}")

        #Find all exits from node
        self._exits, self._exit_dirs = self._find_exits_from_node(self._current_node)
        exit_info = [f"Point: {ex.fixed_phases}, {ex.free_phases}, {ex_dir}" for ex, ex_dir in zip(self._exits, self._exit_dirs)]
        _log.info(f"Found exits {exit_info}")
        self._exit_index = 0

    def _find_exits_from_node(self, node: Node):
        """
        Default exit function which uses the node as an exit if the ExitHint says so

        Derived classes will check if the return of this is empty arrays and if they are, then they"ll fill them up their own way
        """
        exits = []
        exit_dirs = []
        if node.exit_hint == ExitHint.POINT_IS_EXIT:
            _log.info(f"Using node {node.fixed_phases}, {node.free_phases}, {node.global_conditions} as the exit point with directions {node.axis_direction}")
            if node.axis_direction is None:
                exits.append(node)
                exit_dirs.append(None)
            else:
                exits.append(node)
                exit_dirs.append(node.axis_direction)
        return exits, exit_dirs
    
    def _test_direction(self, point: Point, axis_var: v.StateVariable, direction: Direction):
        """
        Given an axis variable and direction, test that we can step from the point
            If stepping doesn"t work, we continue to reduce the step size until it can (or until the step size is too small)

        We check whether stepping is possible by checking if equilibrium converged, if it"s still global min and if the number of phases stayed the same
            We"ll skip the other two checks (axis lims/distance and similar compositions) since technically, these are valid points that can be added
        """
        #For stepping, there is not other variable
        #For tielines and isopleths, we keep track of the other av
        #  This is mainly to check which axis variable changes the least when stepping
        #  Where we want to be stepping in the variable that changes the other variable the least
        if len(self.axis_vars) == 1:
            other_av, other_av_val = None, None
        else:
            other_av = self.axis_vars[1-self.axis_vars.index(axis_var)]
            other_av_val = point.get_property(other_av)

        # Assume that we can step using the specified axis delta,
        # but, if stepping fails, then we reduce the step size until 
        # it works. If the step size becomes too small, then we stop

        #Temporary, we could default the starting delta to be the minimum, then scale up to the
        #input delta after successful zpf line steps. This could avoid some extra equilibrum calcs
        #If this works better then testing and scaling down the direction, then we don't need the while loop
        #  It would also be easier to return a failure mode for when the direction testing failed (non-convergence, not global min, etc)

        #curr_delta = self.axis_delta[axis_var]
        curr_delta = self.axis_delta[axis_var] * self.MIN_DELTA_RATIO
        while curr_delta >= self.axis_delta[axis_var]*self.MIN_DELTA_RATIO:
            #Step condition, if we hit the axis limit, then scale down the delta and try again
            new_conds, hit_axis_limit = self._step_conditions(point, axis_var, curr_delta, self.axis_lims[axis_var], direction)
            if not hit_axis_limit:
                step_results = zeq.update_equilibrium_with_new_conditions(point, new_conds, self._other_av(axis_var))

                extra_args = {
                    "system_info": self.system_info,
                    "pdens": self.GLOBAL_MIN_PDENS,
                    "tol": self.GLOBAL_MIN_TOL
                }

                #Check valid equilibrium, global min and change in phases
                check_functions = [zchk.simple_check_valid_point, zchk.simple_check_change_in_phases, zchk.simple_check_global_min]
                valid_point = True
                for checks in check_functions:
                    if not checks(step_results, **extra_args):
                        valid_point = False
                        break

                #If valid point, then record the change in the other axis and return both deltas
                if valid_point:
                    other_av_delta = None
                    new_point, orig_cs = step_results
                    if other_av is not None:
                        new_other_av_val = new_point.get_property(other_av)
                        other_av_delta = abs(other_av_val - new_other_av_val)
                    return curr_delta, other_av_delta
            
            _log.info(f"Stepping point {point.fixed_phases}, {point.free_phases}, {point.global_conditions}, {axis_var}, {direction} failed. Reducing step size from {curr_delta} -> {curr_delta*self.DELTA_SCALE}")
            curr_delta *= self.DELTA_SCALE
        
        #If stepping failed and we reached the minimum step size, then return None
        _log.info(f"Stepping point {point.fixed_phases}, {point.free_phases}, {point.global_conditions}, {direction} failed with minimum step size {curr_delta}.")
        return None


