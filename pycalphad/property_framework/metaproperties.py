from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy.typing as npt
from pycalphad.core.minimizer import advance_state
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.solver import Solver
if TYPE_CHECKING:
    from pycalphad.core.workspace import Workspace
from pycalphad.property_framework.computed_property import as_property, ComputableProperty
from pycalphad.property_framework import units
import numpy as np
from copy import copy

def find_first_compset(phase_name: str, wks: "Workspace"):
    if phase_name in wks.phases:
        for _, compsets in wks.enumerate_composition_sets():
            for compset in compsets:
                if compset.phase_record.phase_name == phase_name:
                    return compset
    # couldn't find one in the existing workspace; create a single-phase calculation and try again
    copy_wks = wks.copy()
    copy_wks.phases = [phase_name]
    for _, compsets in copy_wks.enumerate_composition_sets():
        for compset in compsets:
            if compset.phase_record.phase_name == phase_name:
                return compset
    return None

class DrivingForce:
    phase_name: str
    implementation_units = units.energy_implementation_units
    display_units = units.energy_display_units
    display_name = 'Driving Force'

    def __init__(self, phase_name):
        self.phase_name = phase_name

    def __str__(self):
        return f'{self.__class__.__name__}({self.phase_name})'

    def expand_wildcard(self, phase_names):
        return [self.__class__(phase_name) for phase_name in phase_names]

    @property
    def shape(self):
        return tuple()

    @property
    def multiplicity(self):
        if self.phase_name is not None:
            tokens = self.phase_name.split('#')
            if len(tokens) > 1:
                return int(tokens[1])
            else:
                return 1
        else:
            return None

    @property
    def phase_name_without_suffix(self):
        if self.phase_name is not None:
            tokens = self.phase_name.split('#')
            return tokens[0]
        else:
            return None

    def filtered(self, input_compsets):
        "Return a generator of CompositionSets applicable to the current property"
        multiplicity_seen = 0

        for cs_idx, compset in enumerate(input_compsets):
            if (self.phase_name is not None) and compset.phase_record.phase_name != self.phase_name_without_suffix:
                continue
            if (compset.NP == 0) and (not compset.fixed):
                continue
            if self.phase_name is not None:
                multiplicity_seen += 1
                if self.multiplicity != multiplicity_seen:
                    continue
            yield cs_idx, compset

    def compute_property(self, compsets: List[CompositionSet], cur_conds: Dict[str, float],
                         chemical_potentials: npt.ArrayLike) -> float:
        driving_force = float('nan')
        seen_phases = 0
        for _, compset in self.filtered(compsets):
            driving_force = np.dot(chemical_potentials, compset.X) - compset.energy
            seen_phases += 1
        if seen_phases > 1:
            raise ValueError('DrivingForce was passed multiple stable valid CompositionSets')
        return driving_force

    def jansson_derivative(self, compsets, cur_conds, chemical_potentials, deltas: "JanssonDerivativeDeltas") -> npt.ArrayLike:
        "Compute Jansson derivative with self as numerator, with the given deltas"
        seen_phases = 0
        jansson_derivative = np.nan
        for cs_idx, compset in self.filtered(compsets):
            if np.isnan(jansson_derivative):
                jansson_derivative = 0.0
            jansson_derivative += np.dot(deltas.delta_chemical_potentials, compset.X)
            deltas_singlephase = copy(deltas)
            deltas_singlephase.delta_sitefracs = [deltas.delta_sitefracs[cs_idx]]
            for el_idx, el in enumerate(compsets[0].phase_record.pure_elements):
                jansson_derivative += chemical_potentials[el_idx] * \
                    as_property('X({0},{1})'.format(self.phase_name, el)).jansson_derivative(compsets, cur_conds, chemical_potentials, deltas)
            jansson_derivative -= as_property('GM({0})'.format(self.phase_name)).jansson_derivative(compsets, cur_conds, chemical_potentials, deltas)
        if seen_phases > 1:
            raise ValueError('DrivingForce was passed multiple stable valid CompositionSets')
        return jansson_derivative

class DormantPhase:
    """
    Meta-property for accessing properties of dormant phases.
    The internal degrees of freedom of a dormant phase are minimized subject to the potentials of the target calculation.
    Dormant phases are not allowed to become stable.
    """
    _compset: CompositionSet
    max_iterations: int = 50

    def __init__(self, phase: Union[CompositionSet, str],
                 wks: Optional["Workspace"]):
        if wks is None:
            if not isinstance(phase, CompositionSet):
                raise ValueError('Dormant phase calculation requires a starting point for the phase;'
                                  ' either a CompositionSet object should be specified, or pass in a Workspace'
                                  ' of a previous calculation including the phase.'
                                )
        if isinstance(phase, CompositionSet):
            compset = phase
        else:
            compset = find_first_compset(phase, wks)  # can be None if there's a convergence failure
        self._compset = compset
        self.solver = Solver()

    def __str__(self):
        return f'{self.__class__.__name__}({self._compset.phase_record.phase_name})'

    def __call__(self, prop: ComputableProperty) -> ComputableProperty:
        prop = as_property(prop)
        class _autoproperty:
            shape = prop.shape
            implementation_units = prop.implementation_units
            display_units = prop.display_units
            display_name = prop.display_name
            @staticmethod
            def compute_property(equilibrium_compsets: List[CompositionSet], cur_conds: Dict[str, float],
                            chemical_potentials: npt.ArrayLike) -> float:
                if self._compset is None:
                    return prop.compute_property([], cur_conds, chemical_potentials)
                state_variables = equilibrium_compsets[0].phase_record.state_variables
                components = equilibrium_compsets[0].phase_record.nonvacant_elements
                # Fix all state variables and chemical potentials
                conditions = {}
                for sv_idx, statevar in enumerate(state_variables):
                    conditions[str(statevar)] = equilibrium_compsets[0].dof[sv_idx]
                for comp_idx, comp in enumerate(components):
                    conditions['MU_'+comp] = chemical_potentials[comp_idx]
                self.solver._fix_state_variables_in_compsets([self._compset], conditions)
                spec = self.solver.get_system_spec([self._compset], conditions)
                state = spec.get_new_state([self._compset])
                state.chemical_potentials[:] = chemical_potentials
                state.recompute(spec)
                converged = False
                for iteration in range(self.max_iterations):
                    state.iteration = iteration
                    converged = spec.check_convergence(state)
                    if converged:
                        break
                    advance_state(spec, state, np.atleast_1d(0.0), 1.0)
                    state.recompute(spec)
                self._compset = state.compsets[0]
                return prop.compute_property([self._compset], cur_conds, chemical_potentials)
            __str__ = lambda _: f'{prop.__str__()} [Dormant({self._compset.phase_record.phase_name})]'
        return _autoproperty()

    @property
    def driving_force(self):
        return self.__call__(DrivingForce(self._compset.phase_record.phase_name))

class IsolatedPhase:
    """
    Meta-property for accessing properties of isolated phases.
    The configuration of an isolated phase is minimized, by itself, subject to the same conditions as the target calculation.
    Other phases (or additional composition sets for the same phase) are not allowed to become stable.
    """
    _compset: CompositionSet

    def __init__(self, phase: Union[CompositionSet, str],
                 wks: Optional["Workspace"]):
        if wks is None:
            if not isinstance(phase, CompositionSet):
                raise ValueError('Isolated phase calculation requires a starting point for the phase;'
                                  ' either a CompositionSet object should be specified, or pass in a Workspace'
                                  ' of a previous calculation including the phase.'
                                )
        if isinstance(phase, CompositionSet):
            compset = phase
        else:
            compset = find_first_compset(phase, wks)  # can be None if there's a convergence failure
        self._compset = compset
        self.solver = Solver()

    def __str__(self):
        return f'{self.__class__.__name__}({self._compset.phase_record.phase_name})'

    def __call__(self, prop: ComputableProperty) -> ComputableProperty:
        prop = as_property(prop)
        class _autoproperty:
            shape = prop.shape
            implementation_units = prop.implementation_units
            display_units = prop.display_units
            display_name = prop.display_name
            @staticmethod
            def compute_property(equilibrium_compsets: List[CompositionSet], cur_conds: Dict[str, float],
                            chemical_potentials: npt.ArrayLike) -> float:
                if self._compset is None:
                    return prop.compute_property([], cur_conds, chemical_potentials)
                self.solver.solve([self._compset], cur_conds)
                return prop.compute_property([self._compset], cur_conds, chemical_potentials)

            @staticmethod
            def jansson_derivative(compsets, cur_conds, chemical_potentials, deltas):
                return prop.jansson_derivative([self._compset], cur_conds, chemical_potentials, deltas)
            __str__ = lambda _: f'{prop.__str__()} [Isolated({self._compset.phase_record.phase_name})]'
        return _autoproperty()

    @property
    def driving_force(self):
        return self.__call__(DrivingForce(self._compset.phase_record.phase_name))


class ReferenceState:
    """
    Meta-property for calculations involving reference states.
    """
    _reference_wks: List["Workspace"]
    _fixed_conds: List
    _floating_conds: List

    def __init__(self,
                 reference_conditions: List[Tuple[str, Dict]],
                 wks: "Workspace"
                ):
        self._reference_wks = []
        for phase_name, ref_conds in reference_conditions:
            new_wks = wks.copy()
            new_wks.phases = [phase_name]
            self._floating_conds = sorted(set(wks.conditions.keys()) - set(ref_conds.keys()), key=str)
            self._fixed_conds = sorted(set(wks.conditions.keys()).intersection(set(ref_conds.keys())), key=str)
            new_wks.conditions = ref_conds
            self._reference_wks.append(new_wks)
        filtered_fixed_conds = []
        for fic in self._fixed_conds:
            if len(set([tuple(rwks.conditions[fic]) for rwks in self._reference_wks])) != 1:
                filtered_fixed_conds.append(fic)
        self._fixed_conds = filtered_fixed_conds
        if len(self._fixed_conds)+1 != len(self._reference_wks):
            raise ValueError('Specified conditions do not define a reference plane')

    def __call__(self, prop: ComputableProperty) -> ComputableProperty:
        prop = as_property(prop)
        class _autoproperty:
            shape = prop.shape
            implementation_units = prop.implementation_units
            display_units = prop.display_units
            display_name = prop.display_name
            @staticmethod
            def compute_property(equilibrium_compsets: List[CompositionSet], cur_conds: Dict[str, float],
                            chemical_potentials: npt.ArrayLike) -> float:
                # Property contribution prior to reference state change
                result = prop.compute_property(equilibrium_compsets, cur_conds, chemical_potentials)

                # Calculate reference contribution

                # First, compute the plane of reference
                plane_matrix = np.zeros((len(self._reference_wks), len(self._fixed_conds)+1))
                # Rightmost column represents the constant term
                plane_matrix[:, -1] = 1
                plane_rhs = np.zeros(len(self._fixed_conds)+1)
                for row_idx, ref_wks in enumerate(self._reference_wks):
                    for col_idx, fic in enumerate(self._fixed_conds):
                        plane_matrix[row_idx, col_idx] = ref_wks.conditions[fic][0]
                    for floc in self._floating_conds:
                        ref_wks.conditions[floc] = cur_conds[floc]
                    if ref_wks.ndim != 0:
                        raise ValueError('Reference state must be point calculation')
                    eq_idx, ref_compsets = list(ref_wks.enumerate_composition_sets())[0]
                    ref_chempots = ref_wks.eq.MU[eq_idx]
                    plane_rhs[row_idx] = prop.compute_property(ref_compsets, {c: val for c, val in ref_wks.conditions.items()}, ref_chempots)
                plane_coefs = np.linalg.solve(plane_matrix, plane_rhs)

                # Next, plug fixed conditions of current point into equation of reference plane
                current_vector = [cur_conds[floc] for floc in self._fixed_conds]
                reference_offset = np.dot(plane_coefs[:-1], current_vector) + plane_coefs[-1]
                return result - reference_offset
            @staticmethod
            def jansson_derivative(equilibrium_compsets, cur_conds, chemical_potentials, deltas):
                # Property contribution prior to reference state change
                result = prop.jansson_derivative(equilibrium_compsets, cur_conds, chemical_potentials, deltas)

                # Calculate reference contribution

                # First, compute the plane of reference
                plane_matrix = np.zeros((len(self._reference_wks), len(self._fixed_conds)+1))
                # Rightmost column represents the constant term
                plane_matrix[:, -1] = 1
                plane_rhs = np.zeros(len(self._fixed_conds)+1)
                for row_idx, ref_wks in enumerate(self._reference_wks):
                    for col_idx, fic in enumerate(self._fixed_conds):
                        plane_matrix[row_idx, col_idx] = ref_wks.conditions[fic][0]
                    for floc in self._floating_conds:
                        ref_wks.conditions[floc] = cur_conds[floc]
                    if ref_wks.ndim != 0:
                        raise ValueError('Reference state must be point calculation')
                    eq_idx, ref_compsets = list(ref_wks.enumerate_composition_sets())[0]
                    ref_chempots = ref_wks.eq.MU[eq_idx]
                    plane_rhs[row_idx] = prop.jansson_derivative(ref_compsets, {c: val for c, val in ref_wks.conditions.items()}, ref_chempots, deltas)
                plane_coefs = np.linalg.solve(plane_matrix, plane_rhs)

                # Next, plug fixed conditions of current point into equation of reference plane
                current_vector = [cur_conds[floc] for floc in self._fixed_conds]
                reference_offset = np.dot(plane_coefs[:-1], current_vector) + plane_coefs[-1]
                return result - reference_offset
            __str__ = lambda _: f'{prop.__str__()} [ReferenceState]'
        return _autoproperty()
