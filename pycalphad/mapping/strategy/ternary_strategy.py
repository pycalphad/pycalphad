from typing import Union
import logging
import itertools
import copy

import numpy as np

from pycalphad import Database, variables as v
from pycalphad.core.composition_set import CompositionSet
from pycalphad.property_framework.computed_property import LinearCombination

from pycalphad.mapping.primitives import ZPFLine, Node, Point, ExitHint, Direction, MIN_COMPOSITION, ZPFState, _eq_compset
from pycalphad.mapping.starting_points import point_from_equilibrium
import pycalphad.mapping.zpf_equilibrium as zeq
import pycalphad.mapping.utils as map_utils
from pycalphad.mapping.strategy.strategy_base import MapStrategy
from pycalphad.mapping.strategy.step_strategy import StepStrategy

_log = logging.getLogger(__name__)

def _get_delta_cs_var(point: Point, comp_sets: list[CompositionSet], axis_vars: list[v.StateVariable], normalize=True):
    """
    Given two composition sets, get the vector using axis_vars as coordinates

    Seems a bit strange to have the comp sets as an input when we already have the point, but it's just to have some
    flexibility on which comp sets to treat at 0 or 1
    """
    dx = point.get_local_property(comp_sets[0], axis_vars[0]) - point.get_local_property(comp_sets[1], axis_vars[0])
    dy = point.get_local_property(comp_sets[0], axis_vars[1]) - point.get_local_property(comp_sets[1], axis_vars[1])
    norm_factor = 1 if not normalize else np.sqrt(dx*dx + dy*dy)
    return [dx/norm_factor, dy/norm_factor]

def _get_norm(point: Point, axis_vars: list[v.StateVariable]):
    """
    Assumes point has two phases
    """
    vec = _get_delta_cs_var(point, point.stable_composition_sets, axis_vars)
    return [-vec[1], vec[0]]

def _create_linear_comb_conditions(point: Point, axis_vars: list[v.StateVariable], normal: list[float] = None):
    #Get normal and axis variable to step in (this will be along the maximum of the normal vector)
    if normal is None:
        normal = _get_norm(point, axis_vars)

    #Create a linear combination that forces the stepping to be along the normal
    c = normal[1]*point.global_conditions[axis_vars[0]] - normal[0]*point.get_property(axis_vars[1])
    lc = LinearCombination(normal[1]*axis_vars[0] - normal[0]*axis_vars[1])
    return lc, c

def _sort_point(point: Point, axis_vars: list[v.StateVariable], norm: dict[v.StateVariable, float]):
    """
    Given a point in a binary system with 2 free phases, get derivative at both composition sets to
    test with CS to fix and which direction to start
    """
    _log.info(f"Sorting point {point.fixed_phases}, {point.free_phases}, {point.global_conditions}")

    #Free all phases in point, we will fix the phase at the end
    free_point = map_utils._generate_point_with_free_cs(point)

    #Get normal and axis variable to step in (this will be along the maximum of the normal vector)
    normal = _get_norm(free_point, axis_vars)
    av = axis_vars[np.argmax(np.abs(normal))]
    _log.info(f"Point {point.fixed_phases}, {point.free_phases} with normal {normal}. Testing derivative in {av}")

    #This is here to have derivative with respect to the axis variable to follow the direction of the normal
    #  As a note here, the derivative we compute is not the directional derivative with respect to the normal
    #  but rather the derivative with respect to the axis variable, with the normal as a fixed condition
    #  Since we lose the direction information of the normal when taking the derivative, mult_factor is here
    #  to retain that information
    mult_factor = np.sign(normal[axis_vars.index(av)])

    #Create a linear combination that forces the stepping to be along the normal
    lc, c = _create_linear_comb_conditions(point, axis_vars, normal)

    #Create temporary set of conditions that replaces the fixed axis variable with the linear combination
    del free_point.global_conditions[axis_vars[1-axis_vars.index(av)]]
    free_point.global_conditions[lc] = c

    #Test phase composition derivative for each phase stepping along the axis variable fixed along the normal
    stable_cs = free_point.stable_composition_sets
    cs_options = [[stable_cs[0], stable_cs[1]], [stable_cs[1], stable_cs[0]]]
    options_tests = []
    for cs_list in cs_options:
        for av_test in axis_vars:
            v_num = v.X(cs_list[1].phase_record.phase_name, av_test.species)
            der = mult_factor*zeq.compute_derivative(free_point, v_num, av, free_den = False)
            new_point = map_utils._generate_point_with_fixed_cs(point, cs_list[0], cs_list[1])
            options_tests.append((der, new_point, av_test))

    #Best index is the one with the largest derivative
    best_index = -1
    best_der = -1
    for i in range(len(options_tests)):
        _log.info(f"Option: Axis var {options_tests[i][2]}, derivative {options_tests[i][0]}, point {options_tests[i][1].fixed_phases}, {options_tests[i][1].free_phases}")
        if abs(options_tests[i][0]) > best_der:
            best_index = i
            best_der = abs(options_tests[i][0])

    if best_index == -1:
        return options_tests[0][0], options_tests[0][1], options_tests[0][2], normal
    else:
        return options_tests[best_index][0], options_tests[best_index][1], options_tests[best_index][2], normal

class TernaryStrategy(MapStrategy):
    def __init__(self, dbf: Database, components: list[str], phases: list[str], conditions: dict[v.StateVariable, Union[float, tuple[float]]], **kwargs):
        super().__init__(dbf, components, phases, conditions, **kwargs)
        #TODO: I don't really like this since this assumes pure elements
        #Although since species as conditions aren't supported yet, I suppose it's fine for now
        unlisted_element = list(set(self.components) - {'VA'} - set([str(av.species) for av in self.axis_vars]))[0]
        self.all_vars = self.axis_vars + [v.X(unlisted_element)]

    def initialize(self):
        """
        Searches axis limits to find starting points

        Here, we do a step mapping along the axis bounds and grab all the nodes
        The nodes of a step map is distinguished from starting points in that they have a parent
        """
        map_kwargs = self._constant_kwargs()

        #Iterate through axis variables, and set conditions to fix axis variable at min only
        for av in self.axis_vars:
            conds = copy.deepcopy(self.conditions)
            conds[av] = np.amin(self.axis_lims[av])
            
            #other_av = self._other_av(av)
            #av_range = np.amax(self.axis_lims[other_av]) - np.amin(self.axis_lims[other_av])
            #conds[other_av] = (self.axis_lims[other_av][0], self.axis_lims[other_av][1], av_range/20)

            #Adjust composition conditions to be slightly above 0 or below 1 for numerical stability
            if isinstance(av, v.X):
                if conds[av] == 0:
                    conds[av] = MIN_COMPOSITION

            #Step map
            step = StepStrategy(self.dbf, self.components, self.phases, conds, **map_kwargs)
            step.initialize()
            step.do_map()
            self._add_starting_points_from_step(step)

        #Additional step where we switch axis conditions to the unlisted variable
        conds = copy.deepcopy(self.conditions)
        conds[self.all_vars[-1]] = MIN_COMPOSITION
        del conds[self.axis_vars[0]]

        #av_range = np.amax(self.axis_lims[self.axis_vars[1]]) - np.amin(self.axis_lims[self.axis_vars[1]])
        #conds[self.axis_vars[1]] = (self.axis_lims[self.axis_vars[1]][0], self.axis_lims[self.axis_vars[1]][1], av_range/20)

        #Step map
        step = StepStrategy(self.dbf, self.components, self.phases, conds, **map_kwargs)
        step.initialize()
        step.do_map()
        self._add_starting_points_from_step(step)

    def _add_starting_points_from_step(self, step: StepStrategy):
        """
        Adds all 2-phase and 3-phase regions from step as starting points
        We also do a global min check to make sure these phase regions are truly the phases they say they are
        """
        for zpf_line in step.zpf_lines:
            if len(zpf_line.stable_phases) == 2 or len(zpf_line.stable_phases) == 3:
                num_phases = len(zpf_line.stable_phases)
                p_index = 0
                while len(zpf_line.points[p_index].stable_phases) != num_phases:
                    p_index += 1
                
                if len(zpf_line.points[p_index].stable_phases) == num_phases:
                    new_point = zpf_line.points[p_index]
                    if self.all_vars[-1] in new_point.global_conditions:
                        del new_point.global_conditions[self.all_vars[-1]]
                        new_point.global_conditions[self.axis_vars[0]] = new_point.get_property(self.axis_vars[0])
                    _log.info(f"Adding node {new_point.fixed_phases}, {new_point.free_phases}, {new_point.global_conditions}")

                    free_point = map_utils._generate_point_with_free_cs(new_point)

                    if self._check_full_global_equilibrium(free_point, add_global_point_if_false = True):
                        new_node = self._create_node_from_point(free_point, None, None, None)
                        self.node_queue.add_node(new_node)

    def _find_exits_from_node(self, node: Node):
        """
        A node on for a binary system has three exits, which are combinations of 2 CS in the node
        Since the node is found from one pair of CS, one of the exits are already accounted for, so
        practically, it's only 2 exits
            However, if there are multiple starting points, a node may be found from multiple zpf lines
            thus a node may only have 1 or even 0 exits - (does this mean some exits are repeated if a node is found twice?)

        I believe this is the same as the binary strategy
        """
        exits, exit_dirs = super()._find_exits_from_node(node)
        if node.exit_hint == ExitHint.POINT_IS_EXIT:
            return exits, exit_dirs
        
        for cs_1, cs_2 in itertools.combinations(node.stable_composition_sets, 2):
            new_conds = {key: node.get_property(key) for key, val in node.global_conditions.items()}

            #Create point with free composition sets (and ignore whether cs_1 or cs_2 is actually fixed)
            #This will be modified in _determine_start_direction
            candidate_point = Point.with_copy(new_conds, node.chemical_potentials, [], [cs_1, cs_2])

            if not node.has_point_been_encountered(candidate_point, False) or node.exit_hint == ExitHint.FORCE_ALL_EXITS:
                _log.info(f"Found candidate exit: {candidate_point.fixed_phases}, {candidate_point.free_phases}, {candidate_point.global_conditions}")
                exits.append(candidate_point)
                #This function is only responsible for finding exit points
                #The _determine_start_direction will take the point and find the direction and axis variable
                exit_dirs.append(None)

        return exits, exit_dirs
    
    def _determine_start_direction(self, node: Node, exit_point: Point, proposed_direction: Direction):
        """
        For stepping, only one direction is possible from a node since we either step positive or negative

        If a direction cannot be found, then we force add a starting point just past the exit_point
        """
        #Sort exit point to fix composition set that varies the least
        norm = {av: self.normalize_factor(av) for av in self.axis_vars}
        der, exit_point, axis_var, normal = _sort_point(exit_point, self.axis_vars, norm)

        free_cs = exit_point.free_composition_sets[0]
        #If node is invariant, then we can get the other cs to know which direction to step away from
        #It's possible that the node is not invariant (ex. as a starting point), in which case, we just set the norm_delta_dot to 1
        if len(node.stable_composition_sets) == 3:
            other_cs = [cs for cs in node.stable_composition_sets if not _eq_compset(cs, exit_point.fixed_composition_sets[0]) and not _eq_compset(cs, exit_point.free_composition_sets[0])][0]
            delta_var = _get_delta_cs_var(exit_point, [free_cs, other_cs], self.axis_vars)
            norm_delta_dot = np.dot(delta_var, normal)
        else:
            norm_delta_dot = 1


        if proposed_direction is None:
            #If the derivative of the axis variable is positive and the normal is the same direction as the delta
            #Or if derivative is negative and normal is opposite direction, then stepping is Positive direction 
            #would be preferred. Same goes for the two other cases where Negative direction is preferred
            #We add the other direction just in case the first proposed direction fails
            if norm_delta_dot * der > 0:
                directions = [Direction.POSITIVE, Direction.NEGATIVE]
            else:
                directions = [Direction.NEGATIVE, Direction.POSITIVE]
        else:
            directions = [proposed_direction]

        for d in directions:
            dir_results = self._test_direction(exit_point, axis_var, d)
            if dir_results is not None:
                av_delta, other_av_delta = dir_results
                _log.info(f"Found direction: {axis_var, d, av_delta} for point {exit_point.fixed_phases}, {exit_point.free_phases}, {exit_point.global_conditions}")
                return exit_point, axis_var, d, av_delta
            
        self._add_starting_point_at_new_condition(exit_point, normal, Direction.POSITIVE if norm_delta_dot > 0 else Direction.NEGATIVE)
            
        return None
    
    def _test_swap_axis(self, zpf_line: ZPFLine):
        """
        By default, we won"t swap axis. This will be the case for stepping
        For more than 2 axis, we do a comparison of how much each axis variable changed in the last two steps

        Same as swapping for binary case
        """
        if len(zpf_line.points) > 1:
            #Get change in axis variable for both variables
            curr_point = zpf_line.points[-1]
            prev_point = zpf_line.points[-2]
            dv = [(curr_point.get_property(av) - prev_point.get_property(av))/self.normalize_factor(av) for av in self.axis_vars]

            #We want to step in the axis variable that changes the most (that way the change in the other variable will be minimal)
            #We also can get the direction from the change in variable
            index = np.argmax(np.abs(dv))
            direction = Direction.POSITIVE if dv[index] > 0 else Direction.NEGATIVE
            if zpf_line.axis_var != self.axis_vars[index]:
                _log.info(f"Swapping axis to {self.axis_vars[index]}. ZPF vector {dv} {self.axis_vars}")
                
                #Since we check the change in axis variable at the current delta, we'll retain the same delta
                #when switching axis variable (same delta as a ratio of the initial delta)
                delta_scale = zpf_line.current_delta / self.axis_delta[zpf_line.axis_var]
                zpf_line.axis_var = self.axis_vars[index]
                zpf_line.axis_direction = direction
                zpf_line.current_delta = self.axis_delta[zpf_line.axis_var] * delta_scale

    def _check_full_global_equilibrium(self, node: Node, add_global_point_if_false = True):
        """
        Does a full global equilibrium calculation to check if point is global min

        Sometimes, a false global min invariant will be added, so this is a final check to make sure an invariant is real
        """
        # We'll bias the composition towards the free composition sets, this can sometimes help with equilibrium (especially if the node
        # is found from different ZPF lines, then we have multiple compositions to check that this node exists)
        # There seems to be some cases where a phase is not detected even when it's supposed to be there
        free_point = map_utils._generate_point_with_free_cs(node, bias_towards_free=True)
        _log.info(f"Checking global eq at {free_point.global_conditions}")
        test_point = point_from_equilibrium(self.dbf, self.components, self.phases, free_point.global_conditions, None, models=self.models, phase_record_factory=self.phase_records)
        if test_point is None:
            return False, None
        
        #If the test point has the same composition sets as the free point, then it's valid node
        if free_point == test_point:
            return True, test_point
        
        #If test point has a different set of phases, then we can add the new point as a starting point
        else:
            if add_global_point_if_false:
                _log.info(f"Global eq check failed. Adding test point as a starting point {test_point.fixed_phases}, {test_point.free_phases}")
                new_node = self._create_node_from_point(test_point, None, None, None)
                self.node_queue.add_node(new_node)
            return False, test_point
    
    def _process_new_node(self, zpf_line: ZPFLine, new_node: Node):
        """
        Some extra checks before adding a new node to the node queue

        1. Check global equilibrium to make sure node is true global eq
            If global eq failed (convergence issue), then decrease step size and try stepping again
              If step size is too small, then fail zpf line and remove it
        2. If passed, then add new node like normal
        3. If node is not true equilibrium, check if the zpf line phases is a subset of the node phases
            If subset, then set the last zpf point as a parent and add new node like normal
            If not, then decrease step size and try stepping again
              If step size is too small, then fail zpf line and add node as a new starting point
        """
        #Do a global equilibrium check before attempting to add the new node
        global_eq_check, test_point = self._check_full_global_equilibrium(new_node, add_global_point_if_false = False)
        if test_point is None:
            _log.info("Global eq check could not be determined")
            #zpf_line.status = ZPFState.FAILED
            #return

            zpf_line.current_delta *= self.DELTA_SCALE
            if zpf_line.current_delta / self.axis_delta[zpf_line.axis_var] < self.MIN_DELTA_RATIO:
                zpf_line.status = ZPFState.FAILED

                #Removes the last ZPF line if we determine that the new node is not connected to the current zpf line
                #Since the last zpf line has finished (or is non-existent if this is the first zpf line), this will
                #cause the iteration to go directly to the next node or exit
                self.zpf_lines.pop(-1)
            else:
                zpf_line.status = ZPFState.NOT_FINISHED

        else:
            if global_eq_check:
                _log.info("Global eq check passed")
                return super()._process_new_node(zpf_line, new_node)
            else:
                zpf_line_phases = zpf_line.stable_phases_with_multiplicity
                test_node_phases = test_point.stable_phases_with_multiplicity
                _log.info(f"Global eq check failed. Comparing zpf line {zpf_line_phases} with node phases {test_node_phases}")
                if len(set(test_node_phases) - set(zpf_line_phases)) == 1:
                    _log.info("Node can be added as a proper node")
                    new_node = self._create_node_from_point(test_point, zpf_line.points[-1], None, None)
                    return super()._process_new_node(zpf_line, new_node)
                else:
                    #zpf_line.status = ZPFState.FAILED
                    #return
                    zpf_line.current_delta *= self.DELTA_SCALE
                    if zpf_line.current_delta / self.axis_delta[zpf_line.axis_var] < self.MIN_DELTA_RATIO:
                        zpf_line.status = ZPFState.FAILED
                        _log.info("Node is added as a starting point")
                        new_node = self._create_node_from_point(test_point, None, None, None)
                        if not self.node_queue.add_node(new_node):
                            _log.info(f"Node {new_node.fixed_phases}, {new_node.free_phases} has already been added")
                        self.zpf_lines.pop(-1)
                    else:
                        zpf_line.status = ZPFState.NOT_FINISHED

    def _add_starting_point_at_new_condition(self, point: Point, normal: list[float], direction: Direction):
        # If we made it here, then no direction has worked. This could be a case where
        # the two CS of the exit point are stoichiometric, so there is no ZPF line leading
        # to the next node
        # So we take a step from the normal of the exit away from the third CS of the invariant
        # and add a new starting point. We also add the current exit as a parent, so we don't
        # search in that direction again
        free_point = map_utils._generate_point_with_free_cs(point)
        copy_conds = copy.deepcopy(free_point.global_conditions)
        #Move by small amount (1e-3 seem to be a good value, too small, and we may fail to detect a possible third phase and too large, and we may step over a potential node)
        for av, norm_dir in zip(self.axis_vars, normal):
            copy_conds[av] += 1e-3*norm_dir*direction.value

        _log.info(f"Attemping to add point at {copy_conds}")
        new_point = point_from_equilibrium(self.dbf, self.components, self.phases, copy_conds, None, models=self.models, phase_record_factory=self.phase_records)
        if new_point is not None:
            #If new point is an invariant, then we add it as a starting point (point is added as a parent to prevent repeated exit calculations)
            #If new point is 2 phase, then we add it as a starting point the node being the exit
            _log.info(f"Adding starting point: {new_point.stable_phases}")
            success = False
            if len(new_point.stable_composition_sets) == 3:
                new_node = self._create_node_from_point(new_point, point, None, None)
                success = self.node_queue.add_node(new_node)
            elif len(new_point.stable_composition_sets) == 2:
                new_node = self._create_node_from_point(new_point, None, None, None, ExitHint.POINT_IS_EXIT)
                success = self.node_queue.add_node(new_node)
            else:
                _log.info(f"Point could not be starting point. Needs 2 or 3 phases")
            if not success:
                _log.info(f"Point has already been added")
        else:
            _log.info(f"Could not find a new node from conditions")

            

