from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List, Mapping
import numpy as np

from pycalphad import variables as v
from pycalphad.core.composition_set import CompositionSet

# must be in sorted order
STATEVARS = [v.N, v.P, v.T]  # TODO: global, will change once pycalphad drops strict v.N condition
CS_EQ_TOL = 1e-8
MIN_COMPOSITION = 1e-6

class Direction(Enum):
    NEGATIVE = -1
    POSITIVE = +1

class ExitHint(Enum):
    """
    Exit rules

    NORMAL - will search for all viable exits from node
             ignores the exit that corresponds to the ZPF line that found the node
    POINT_IS_EXIT - will use the point composition sets as the exit
                    this is mainly used for starting points since the point may not necessary fit the conditions for a "node" (0 DOF)
    FORCE_ALL_EXITS - similar to normal, but will include every possible exit from node
                      not sure what the use case for this is, seems to have been for ad-hoc fix that is no longer needed?
                      Only thing I can think of is for a starting point on a node for tielines or isopleths strategies
    """
    NORMAL = 0
    POINT_IS_EXIT = 1
    FORCE_ALL_EXITS = 2

def _eq_compset(compset: CompositionSet, other: CompositionSet):
    """
    2 tests
        If compset and other point are the same object (same memory)
        If compset and other has the same phase name and DOF (within tolerance)
    """
    # Composition set equality
    if compset is other:
        return True
    if compset.phase_record.phase_name != other.phase_record.phase_name:
        return False
    if np.allclose(compset.dof, other.dof, atol=CS_EQ_TOL):
        return True
    return False

def _get_phase_list_with_multiplicity(phases: list[str]):
    """
    Helper function to get unique list of phases
    If a miscibility gap is present in the phase list, this will add a #n to the phase name
    """
    u_phases = []
    for p in phases:
        test_name = p
        phase_id = 2
        while test_name in u_phases:
            test_name = f'{p}#{phase_id}'
            phase_id += 1
        u_phases.append(test_name)
    return u_phases

def _get_phase_specific_variable(phase: str, var: v.StateVariable, is_global = False):
    """
    Helper function for ZPFLine.get_var_list

    Converts variable to phase specific if possible

    If variable is a state variable or non-phase dependent, then return the same variable
    For variables such as x or NP, then we return x or NP of phase
        If we specify that the variable is global (for x), then we return x unchanged
    """
    if is_global:
        return var
    if isinstance(var, v.X):
        return v.X(phase, var.species)
    elif isinstance(var, v.NP) or var == v.NP:
        return v.NP(phase)
    else:
        return var

@dataclass
class Point():
    """
    Stores data for a single point on the map
    This will include everything needed to compute any property from pycalphad workspace
        composition sets, conditions, chemical potentials

    Fixed and free composition sets are split for easy accounting
    """
    global_conditions: Mapping[v.StateVariable, float]
    chemical_potentials: List[float]    #We"ll store chemical potentials in case someone wants to plot activity diagrams

    # Yes, this uses CompositionSet objects, which means that there are copies saved.
    # Maybe inefficient, but we _need_ to know phase compositions for plotting and site
    # fractions are nice to have for more detailed reconstruction and post-processing.
    _fixed_composition_sets: List[CompositionSet]
    _free_composition_sets: List[CompositionSet]

    # Note: The following three functions make a shallow copy of the composition sets
    @property
    def stable_composition_sets(self):
        return self._fixed_composition_sets + self._free_composition_sets

    @property
    def stable_composition_sets_flipped(self):
        return self._free_composition_sets + self._fixed_composition_sets

    @property
    def fixed_composition_sets(self) -> List[CompositionSet]:
        return [cs for cs in self.stable_composition_sets if cs.fixed]

    @property
    def free_composition_sets(self) -> List[CompositionSet]:
        return [cs for cs in self.stable_composition_sets if not cs.fixed]

    @property
    def stable_phases(self):
        return [cs.phase_record.phase_name for cs in self.stable_composition_sets]

    @property
    def fixed_phases(self):
        return [cs.phase_record.phase_name for cs in self.fixed_composition_sets]

    @property
    def free_phases(self):
        return [cs.phase_record.phase_name for cs in self.free_composition_sets]

    @property
    def stable_phases_with_multiplicity(self):
        return _get_phase_list_with_multiplicity(self.stable_phases)

    @property
    def fixed_phases_with_multiplicity(self):
        return _get_phase_list_with_multiplicity(self.fixed_phases)

    @property
    def free_phases_with_multiplicity(self):
        return _get_phase_list_with_multiplicity(self.free_phases)

    # Creates a deep copy of the point
    def create_copy(self):
        return deepcopy(self)

    # Creates point with deep copy of inputs (composition sets, conditions, chemical potentials)
    @classmethod
    def with_copy(cls, *args, **kwargs):
        return cls(*deepcopy(args), **deepcopy(kwargs))

    def __eq__(self, other):
        """
        This equality treats all composition sets as the same (regardless of wether it"s fixed or not)
            So this also ignores phase fraction, but we"re just checking if the phase boundaries
            are the same
        """
        if not isinstance(other, Point):
            return False
        # Nodes are special points that represent where a set of phases change
        # Thus it is useful to be able to say if two nodes are equal, in that they
        # correspond to the same point. Only the stable set of phases need to be the same
        # fixed or free doesn"t matter.
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
        """
        This equality accounts for fixed composition sets so we can check if two points are exactly the same
        This also ignores phase fraction
        """
        if self == other:
            # Compare fixed composition sets
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
        output = "Fixed CS: " + str([c.phase_record.phase_name for c in self.fixed_composition_sets])
        output += "\nFree CS: " + str([c.phase_record.phase_name for c in self.free_composition_sets])
        output += "\nConditions: " + str(self.global_conditions)
        output += "\nChem_pot: " + str(self.chemical_potentials)
        return output

    def get_property(self, var: v.StateVariable):
        """
        Wrapper around compute property so I don't have long lines of code getting composition sets, conditions and chemical potentials everywhere
        We will also squeeze the results since v.MoleFraction seems to return an array
        """
        return np.squeeze(var.compute_property(self.stable_composition_sets, self.global_conditions, self.chemical_potentials))

    def get_local_property(self, comp_set: CompositionSet, var: v.StateVariable):
        """
        Another wrapper around compute property, this time, it is applied to a single composition set

        We take the assumption here that NP = 1, so we have to correct for v.X and v.NP
        """
        # Store current phase fraction. Easiest way to make the NP=1 assumption is the literally make NP=1
        curr_np = comp_set.NP
        # Can't use map utils here due to circular dependencies
        comp_set.update(comp_set.dof[len(STATEVARS):], 1.0, comp_set.dof[:len(STATEVARS)])

        prop_value = var.compute_property([comp_set], self.global_conditions, self.chemical_potentials)

        # Restore phase fraction
        comp_set.update(comp_set.dof[len(STATEVARS):], curr_np, comp_set.dof[:len(STATEVARS)])

        return np.squeeze(prop_value)

@dataclass(eq=False)
class Node(Point):
    """
    A Node is a special case of a Point that indicates a set of conditions
    corresponding to a phase change.

    It also keeps track of the parent Point that it was created from, so that
    users of the class can determine which exits from the node are novel.

    Compared to its parent, the set of conditions should be the same (both keys
    and values) and the set of stable phases should differ by exactly one phase.

    We"ll keep track of the axis variable and direction as well
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

    def __str__(self):
        output = super().__str__()
        output += "\nAxis: " + str([self.axis_var, self.axis_direction])
        return output

class ZPFState(Enum):
    """
    NOT_FINISHED - zpf line is not finished
    GRACEFUL_ENDING - zpf line ended properly (new node is found or it reached axis limits)
    UNEXPECTED_ENDING - zpf line ended improperly (error in equilibrium calculation, couldn"t find new node)

    This is to track whether a zpf line ends prematurely, and if so, we may attempt to force another starting point
    """
    NOT_FINISHED = 0
    NEW_NODE_FOUND = 1
    REACHED_LIMIT = 2
    FAILED = 3
    ATTEMPT_NEW_STEP = 4

class ZPFLine():
    """
    ZPF line represents a line where a phase change occur (crossing the bounding will add or remove a phase, and that phase is 0 at the boundary)
        Number of phases is constant along this line
    Defines a list of fixed phases (the zero phases) and list of free phases and list of Point that represents the line

    finished variable will tell the mapper whether the zpf line is finished to start the next zpf line
    """
    def __init__(self, fixed_phases : List[str], free_phases : List[str]):
        # Miscibility gaps should have duplicate entries
        self.fixed_phases: List[str] = fixed_phases
        self.free_phases: List[str] = free_phases
        self.points: List[Point] = []
        self.last_globally_checked_index: int = 0
        self.status: ZPFState = ZPFState.NOT_FINISHED
        self.axis_var: v.StateVariable = None
        self.axis_direction: Direction = None
        self.current_delta: float = 1

    @property
    def stable_phases(self):
        return self.fixed_phases + self.free_phases

    @property
    def stable_phases_with_multiplicity(self):
        return _get_phase_list_with_multiplicity(self.stable_phases)

    @property
    def fixed_phases_with_multiplicity(self):
        return _get_phase_list_with_multiplicity(self.fixed_phases)

    @property
    def free_phases_with_multiplicity(self):
        return _get_phase_list_with_multiplicity(self.free_phases)

    def num_fixed_phases(self):
        return len(self.fixed_phases)

    def num_free_phases(self):
        return len(self.free_phases)

    def append(self, point: Point):
        self.points.append(point)

    def __str__(self):
        output = str(self.free_phases) + " " + str(self.fixed_phases) + " " + str(len(self.points)) + " " + str(self.points[0].global_conditions) + " " + str(self.points[-1].global_conditions)
        return output

    def get_var_list(self, var : v.StateVariable):
        """
        Gets variable along ZPF line and returns list

        The variables will decipher between local and global variables
        """
        return np.array([p.get_property(var) for p in self.points])

class NodesExhaustedError(Exception):
    pass

class NodeQueue():
    """
    Not exactly a queue, but functions as a queue with adding and getting node in FIFO order
        If we were to use a Queue, we would have to compare the current Node with a buffer stored in the mapper to check for repeating nodes
        So using a List allows the Node checking implementation to be here instead

    When getting a node, it will return the node at the current node index and increase the index counter
        Queue is empty once the index counter points to the last node
        As we don"t remove any nodes from the list (partly because we have to check for repeated nodes), we"ll assume this will be small enough to not cause a huge issue with memory (probably the case unless we plan on doing something weird)
    """
    def __init__(self):
        self.nodes: list[Node] = []
        self._current_node_index = 0

    def add_node(self, candidate_node: Node, force = False) -> bool:
        """
        Checks candidate_node to see if it has been added before
            If it has been added before, add parent to the encountered points list in the node
            When we have multiple start points, we have the chance of encountering a node from multiple ZPF lines
                By keeping a list of all points that lead to this node, we can reduce the number of exits to avoid double calculating ZPF lines

        Force will force add candidate_node, this is useful for starting the zpf line within a two-phase field
        """
        if force:
            self.nodes.append(candidate_node)
            return True
        else:
            # If node is already in node queue, then add the parent to the list
            # of encountered points in the node
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
            raise NodesExhaustedError("No unprocessed nodes remain")

    def size(self):
        """
        Length of the node queue will be how many nodes are left
        """
        return max([len(self.nodes) - self._current_node_index, 0])

    def is_empty(self):
        """
        Since this isn"t a true queue, we can track if the node queue is "empty" by check if the
        current node index reaches the length of the node list

        This is just so we don"t have to raise an exception when the queue is empty
        """
        return len(self.nodes) <= self._current_node_index