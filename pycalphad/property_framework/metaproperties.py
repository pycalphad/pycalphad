from typing import Dict, List, Optional, Union, TYPE_CHECKING
import numpy.typing as npt
from pycalphad.core.minimizer import advance_state
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.solver import Solver
if TYPE_CHECKING:
    from pycalphad.core.workspace import Workspace
from pycalphad.property_framework.computed_property import as_property, ComputableProperty
import numpy as np

def find_first_compset(phase_name: str, wks: "Workspace"):
    for _, compsets in wks.enumerate_composition_sets():
        for compset in compsets:
            if compset.phase_record.phase_name == phase_name:
                return compset
    return None

class DrivingForce:
    phase_name: str

    def __init__(self, phase_name):
        self.phase_name = phase_name

    def __str__(self):
        return f'{self.__class__.__name__}({self.phase_name})'

    def expand_wildcard(self, phase_names):
        return [self.__class__(phase_name) for phase_name in phase_names]

    @property
    def shape(self):
        return (1,)

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
        if seen_phases > 1:
            raise ValueError('DrivingForce was passed multiple stable valid CompositionSets')
        return driving_force

class DormantPhase:
    """
    Meta-property for accessing properties of dormant phases.
    The configuration of a dormant phase is minimized subject to the potentials of the target calculation.
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
        if not isinstance(phase, CompositionSet):
            phase_orig = phase
            phase = find_first_compset(phase, wks)
            if phase is None:
                raise ValueError(f'{phase_orig} is never stable in the specified Workspace')
        self._compset = phase
        self.solver = Solver()

    def __str__(self):
        return f'{self.__class__.__name__}({self._compset.phase_record.phase_name})'

    def __call__(self, prop: ComputableProperty) -> ComputableProperty:
        prop = as_property(prop)
        class _autoproperty:
            shape = prop.shape
            @staticmethod
            def compute_property(equilibrium_compsets: List[CompositionSet], cur_conds: Dict[str, float],
                            chemical_potentials: npt.ArrayLike) -> float:
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
        if not isinstance(phase, CompositionSet):
            phase_orig = phase
            phase = find_first_compset(phase, wks)
            if phase is None:
                raise ValueError(f'{phase_orig} is never stable in the specified Workspace')
        self._compset = phase
        self.solver = Solver()

    def __str__(self):
        return f'{self.__class__.__name__}({self._compset.phase_record.phase_name})'

    def __call__(self, prop: ComputableProperty) -> ComputableProperty:
        prop = as_property(prop)
        class _autoproperty:
            shape = prop.shape
            @staticmethod
            def compute_property(equilibrium_compsets: List[CompositionSet], cur_conds: Dict[str, float],
                            chemical_potentials: npt.ArrayLike) -> float:
                self.solver.solve([self._compset], cur_conds)
                return prop.compute_property([self._compset], cur_conds, chemical_potentials)
            __str__ = lambda _: f'{prop.__str__()} [Isolated({self._compset.phase_record.phase_name})]'
        return _autoproperty()

    @property
    def driving_force(self):
        return self.__call__(DrivingForce(self._compset.phase_record.phase_name))
