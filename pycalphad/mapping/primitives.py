from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List, Mapping
import numpy as np
from pycalphad import variables as v
from pycalphad.core.composition_set import CompositionSet
import matplotlib.pyplot as plt

# must be in sorted order
STATEVARS = [v.N, v.P, v.T]  # TODO: global, will change once pycalphad drops strict v.N condition
CS_EQ_TOL = 1e-8

class Direction(Enum):
    NEGATIVE = -1
    POSITIVE = +1

class ExitHint(Enum):
    NORMAL = 0
    NO_EXITS = 1
    FORCE_ALL_EXITS = 2

def _eq_compset(compset: CompositionSet, other: CompositionSet):
    # Composition set equality
    if compset is other:
        return True
    if compset.phase_record.phase_name != other.phase_record.phase_name:
        return False
    if np.allclose(compset.dof, other.dof, atol=CS_EQ_TOL):
        return True
    return False

@dataclass
class Point():
    global_conditions: Mapping[v.StateVariable, float]
    # Yes, this uses CompositionSet objects, which means that there are copies saved.
    # Maybe inefficient, but we _need_ to know phase compositions for plotting and site
    # fractions are nice to have for more detailed reconstruction and post-processing.
    _fixed_composition_sets: List[CompositionSet]
    _free_composition_sets: List[CompositionSet]
    metastable_composition_sets: List[CompositionSet]  # private, for efficient, reuse purposes

    @property
    def stable_composition_sets(self):
        return self._fixed_composition_sets + self._free_composition_sets

    @property
    def fixed_composition_sets(self) -> List[CompositionSet]:
        return [cs for cs in self.stable_composition_sets if cs.fixed]

    @property
    def free_composition_sets(self) -> List[CompositionSet]:
        return [cs for cs in self.stable_composition_sets if not cs.fixed]

    def create_copy(self):
        return deepcopy(self)

    @classmethod
    def with_copy(cls, *args, **kwargs):
        return cls(*deepcopy(args), **deepcopy(kwargs))

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        # Nodes are special points that represent where a set of phases change
        # Thus it is useful to be able to say if two nodes are equal, in that they
        # correspond to the same point. Only the stable set of phases need to be the same
        # fixed or free doesn't matter.
        if len(self.stable_composition_sets) != len(other.stable_composition_sets):
            return False
        for self_cs in self.stable_composition_sets:
            for other_cs in other.stable_composition_sets:
                if _eq_compset(self_cs, other_cs):
                    # found a match, done for this compset
                    break
            else:
                # We made it through the loop of other compsets with no matches
                return False
        return True

    def compare_consider_fixed_cs(self, other):
        if self == other:
            #Compare fixed composition sets
            if len(self.fixed_composition_sets) != len(other.fixed_composition_sets):
                return False
            for self_cs in self.fixed_composition_sets:
                for other_cs in other.fixed_composition_sets:
                    if _eq_compset(self_cs, other_cs):
                        break
                else:
                    return False
            return True
        else:
            return False

    def __str__(self):
        output = str([c.phase_record.phase_name for c in self.fixed_composition_sets])
        output += str([c.phase_record.phase_name for c in self.free_composition_sets])
        output += str(self.global_conditions)
        return output


@dataclass(eq=False)
class Node(Point):
    """
    A Node is a special case of a Point that indicates a set of conditions
    corresponding to a phase change.

    It also keeps track of the parent Point that it was created from, so that
    users of the class can determine which exits from the node are novel.

    Compared to its parent, the set of conditions should be the same (both keys
    and values) and the set of stable phases should differ by exactly one phase.

    We'll keep track of the axis variable and direction as well
        By default, they will be None and the direction will be decided later
    """
    parent: Point
    axis_var: v.StateVariable = None
    axis_direction: Direction = None
    exit_hint: ExitHint = ExitHint.NORMAL

    def __post_init__(self):
        self.encountered_points = [self.parent]

    def has_point_been_encountered(self, point : Point, test_fixed = False):
        if test_fixed:
            for other in self.encountered_points:
                if point.compare_consider_fixed_cs(other):
                    return True
            return False
        else:
            for other in self.encountered_points:
                if point == other:
                    return True
            return False

    def start_direction(self):
        return (self.axis_var, self.axis_direction)

    def __str__(self):
        output = str([c.phase_record.phase_name for c in self.fixed_composition_sets])
        output += str([c.phase_record.phase_name for c in self.free_composition_sets])
        output += str(self.global_conditions)
        output += str([self.axis_var, self.axis_direction])
        return output

class ZPFLine():
    '''
    ZPF line represents a line where a phase change occur (crossing the bounding will add or remove a phase, and that phase is 0 at the boundary)
        Number of phases is constant along this line
    Defines a list of fixed phases (the zero phases) and list of free phases and list of Point that represents the line

    finished variable will tell the mapper whether the zpf line is finished to start the next zpf line
    '''
    def __init__(self, fixed_phases, free_phases):
        # Miscibility gaps should have duplicate entries
        self.fixed_phases: List[str] = fixed_phases
        self.free_phases: List[str] = free_phases
        self.points: List[Point] = []
        self.last_globally_checked_index: int = 0
        self.finished: bool = False
        self.axis_var: v.StateVariable = None
        self.axis_direction: Direction = None
        self.current_delta: float = 1

    def num_fixed_phases(self):
        return len(self.fixed_phases)

    def num_free_phases(self):
        return len(self.free_phases)

    def append(self, point: Point):
        self.points.append(point)

    def __str__(self):
        output = str(self.free_phases) + ' ' + str(self.fixed_phases) + ' ' + str(len(self.points)) + ' ' + str(self.points[0].global_conditions) + ' ' + str(self.points[-1].global_conditions)
        return output

    def unique_phases(self):
        u_phases = []
        total_phases = self.fixed_phases + self.free_phases
        for p in total_phases:
            test_name = p
            phase_id = 1
            while test_name in u_phases:
                test_name = p + '#' + str(phase_id)
                phase_id += 1
            u_phases.append(test_name)
        return u_phases

    def get_global_var_list(self, var):
        '''
        Gets global variable along ZPF line and returns list
        '''
        x = []
        for p in self.points:
            x.append(_get_global_value_for_var(p, var))
        return x

    def get_global_condition_var_list(self, var):
        x = []
        for p in self.points:
            x.append(p.global_conditions[var])
        return x

    def get_local_var_list(self, var):
        '''
        Gets local variable for each CompositionSet along ZPF line and returns list
        '''
        cs_x = {}
        i = 0       #Counter for current point, used for tracking is phases in a miscibility gap has been added or not
        for p in self.points:
            for cs in p.stable_composition_sets:
                #For NP (phase fraction), ignore input phase and replace with phase name of current composition set
                if isinstance(var, v.NP):
                    p_specific_var = v.NP(cs.phase_record.phase_name)
                    value = _get_value_for_var(cs, p_specific_var)
                else:
                    value = _get_value_for_var(cs, var)

                test_name = cs.phase_record.phase_name
                #If the phase name is already in cs_x, 2 possibilities
                #   The second composition set (from miscibility gap) is not added - adjust name and add to cs_x
                #   The second composition set is already added, but the new point has not been added yet (find first occurence of phase name and add point variable)
                if test_name in cs_x:
                    phase_id = 0
                    #If value has already been added to composition set, then move to next phase_id
                    while len(cs_x[test_name]) == i+1:
                        phase_id += 1
                        test_name = cs.phase_record.phase_name + '#' + str(phase_id)
                        #If phase_id is not in cs_x, then break and add
                        if test_name not in cs_x:
                            cs_x[test_name] = []
                            break
                    cs_x[test_name].append(value)
                else:
                    cs_x[test_name] = [value]
            i += 1

        return cs_x

class NodesExhaustedError(Exception):
    pass

class NodeQueue():
    '''
    Not exactly a queue, but functions as a queue with adding and getting node in FIFO order
        If we were to use a Queue, we would have to compare the current Node with a buffer stored in the mapper to check for repeating nodes
        So using a List allows the Node checking implementation to be here instead

    When getting a node, it will return the node at the current node index and increase the index counter
        Queue is empty once the index counter points to the last node
        As we don't remove any nodes from the list (partly because we have to check for repeated nodes), we'll assume this will be small enough to not cause a huge issue with memory (probably the case unless we plan on doing something weird)
    '''
    def __init__(self):
        self.nodes = []
        self._current_node_index = 0

    def add_node(self, candidate_node: Node, force = False) -> bool:
        '''
        Checks candidate_node to see if it has been added before
            If it has been added before, add parent to the encountered points list in the node
            When we have multiple start points, we have the chance of encountering a node from multiple ZPF lines
                By keeping a list of all points that lead to this node, we can reduce the number of exits to avoid double calculating ZPF lines

        Force will force add candidate_node, this is useful for starting the zpf line within a two-phase field
        '''
        if force:
            self.nodes.append(candidate_node)
            return True
        else:
            #Search t
            for other in self.nodes:
                if other == candidate_node:
                    other.encountered_points.append(candidate_node.parent)
                    return False
            else:
                self.nodes.append(candidate_node)
                return True

    def get_next_node(self):
        if len(self.nodes) > self._current_node_index:
            next_node = self.nodes[self._current_node_index]
            self._current_node_index += 1
            return next_node
        else:
            raise NodesExhaustedError(f"No unprocessed nodes remain")


def _get_value_for_var(compset: CompositionSet, var):
    '''
    For a given variable, extract the value from the composition set
    '''
    if var in STATEVARS:
        return compset.dof[STATEVARS.index(var)]
    elif isinstance(var, v.X):
        # assumes species name is a pure element
        dep_comp_idx = compset.phase_record.nonvacant_elements.index(var.species.name)
        return compset.X[dep_comp_idx]
    elif isinstance(var, v.NP):
        if var.phase_name == compset.phase_record.phase_name:
            return compset.NP
        else:
            return np.nan
    else:
        raise NotImplementedError(f"Variable {var} cannot yet be obtained from CompositionSet objects")


def _get_global_value_for_var(point: Point, var):
    '''
    For a given variable, extract the value of the global condition

    For composition, we have to take it from the weighted composition average for each CompositionSet
    '''
    if var in STATEVARS:
        return point.global_conditions[var]
    elif isinstance(var, v.X):
        # Assumes species name is a pure element
        # For stepping, it is possible that no phases are fixed, so we want to grab the
        # composition index from any stable composition set instead. We sum over the
        # stable composition sets anyways, so it shouldn't be an issue
        dep_comp_idx = point.stable_composition_sets[0].phase_record.nonvacant_elements.index(var.species.name)
        value = 0.0
        for cs in (point.stable_composition_sets):
            value += cs.X[dep_comp_idx] * cs.NP
        return value
    else:
        raise NotImplementedError(f"Variable {var} cannot yet be obtained from Point objects")
