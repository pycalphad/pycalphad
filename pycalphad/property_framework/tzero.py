from typing import Dict, List, Optional, Tuple, Union
import numpy.typing as npt
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.workspace import Workspace
from pycalphad.core.solver import Solver
from pycalphad.property_framework import as_property, DotDerivativeComputedProperty, ConditionableComputableProperty, \
    ModelComputedProperty

def find_first_compset(phase_name: str, wks: Workspace):
    for _, compsets in wks.enumerate_composition_sets():
        for compset in compsets:
            if compset.phase_record.phase_name == phase_name:
                return compset
    return None

class T0(object):
    "T0: (GM(ONE) - GM(TWO))**2 = 0"
    _phase_one: CompositionSet
    _phase_two: CompositionSet
    solver: Solver
    property_to_optimize: ConditionableComputableProperty = as_property('T')
    minimum_value: float = 298.15
    maximum_value: float = 6000
    residual_tol: float = 0.01
    maximum_iterations: int = 50

    def __init__(self, phase_one: Union[CompositionSet, str], phase_two: Union[CompositionSet, str],
                 wks: Optional[Workspace]):
        if wks is None:
            if not isinstance(phase_one, CompositionSet) and not isinstance(phase_two, CompositionSet):
                raise ValueError('T0 calculation requires a starting point for both phases;'
                                  ' either CompositionSet objects should be specified, or pass in a Workspace'
                                  ' of a previous calculation including the phases.'
                                )
        if not isinstance(phase_one, CompositionSet):
            phase_one = find_first_compset(phase_one, wks)
            if phase_one is None:
                raise ValueError(f'{phase_one} is never stable in the specified Workspace')
        if not isinstance(phase_two, CompositionSet):
            phase_two = find_first_compset(phase_two, wks)
            if phase_two is None:
                raise ValueError(f'{phase_two} is never stable in the specified Workspace')
        self._phase_one = phase_one
        self._phase_two = phase_two
        self.solver = Solver()

    def __str__(self):
        return f'{self.__class__.__name__}({self._phase_one.phase_record.phase_name},{self._phase_two.phase_record.phase_name})'

    @property
    def shape(self) -> Tuple[int]:
        return (1,)

    def compute_property(self, equilibrium_compsets: List[CompositionSet], cur_conds: Dict[str, float],
                         chemical_potentials: npt.ArrayLike) -> float:
        s = self.solver
        initial_conditions = cur_conds
        property_to_optimize = str(self.property_to_optimize)

        # T0: (G(BCC) - G(HCP))**2 = 0
        # G(BCC)**2 - 2*G(BCC)*G(HCP) + G(BCP)**2
        # grad = 2*G(BCC)*G'(FCC) - 2*(G'(BCC)*G(HCP) + G'(HCP)*G(BCC)) + 2*G(HCP)*G'(HCP)
        gm_one = ModelComputedProperty('GM', self._phase_one.phase_record.phase_name)
        gm_one_grad = DotDerivativeComputedProperty(gm_one, self.property_to_optimize)
        gm_two = ModelComputedProperty('GM', self._phase_two.phase_record.phase_name)
        gm_two_grad = DotDerivativeComputedProperty(gm_two, self.property_to_optimize)
        conditions = initial_conditions.copy()
        for _ in range(self.maximum_iterations):
            one_result = s.solve([self._phase_one], conditions)
            two_result = s.solve([self._phase_two], conditions)
            one_gm = gm_one.compute_property([self._phase_one], conditions, one_result.chemical_potentials)
            one_grad = gm_one_grad.compute_property([self._phase_one], conditions, one_result.chemical_potentials)

            two_gm = gm_two.compute_property([self._phase_two], conditions, two_result.chemical_potentials)
            two_grad = gm_two_grad.compute_property([self._phase_two], conditions, two_result.chemical_potentials)
            t0_grad = 2*one_gm*one_grad - 2*(one_grad*two_gm + two_grad*one_gm) + 2*two_gm*two_grad

            residual = (one_gm-two_gm)**2
            if abs(t0_grad) < 1e-10:
                t0_step = 0
            else:
                t0_step = -residual/t0_grad
            #print('Step: ', t0_step)
            conditions[property_to_optimize] = max(min(conditions[property_to_optimize] + t0_step[0],
                                                            self.maximum_value),
                                                        self.minimum_value)
            if residual < self.residual_tol:
                break
        if residual > self.residual_tol:
            return float('nan')
        return conditions[property_to_optimize]