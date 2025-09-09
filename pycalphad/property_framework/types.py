from dataclasses import dataclass
from pint._typing import UnitLike
import numpy.typing as npt
from typing import Any, Dict, List, Optional, Tuple
try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable
from pycalphad.core.composition_set import CompositionSet

@dataclass
class JanssonDerivativeDeltas:
    delta_chemical_potentials: Optional[Any]
    delta_statevars: Optional[Any]
    delta_parameters: Optional[Any]
    delta_phase_amounts: Optional[Any]
    delta_sitefracs: Optional[Any]

@runtime_checkable
class ComputableProperty(Protocol):
    implementation_units: UnitLike
    display_units: UnitLike

    def compute_property(self, compsets: List[CompositionSet], cur_conds: Dict[str, float], chemical_potentials: npt.ArrayLike) -> npt.ArrayLike:
        ...
    @property
    def shape(self) -> Tuple[int]:
        ...

@runtime_checkable
class DifferentiableComputableProperty(ComputableProperty, Protocol):
    "Can be in the numerator of a Jansson derivative"
    def jansson_derivative(self, compsets: List[CompositionSet], cur_conds: Dict[str, float], chemical_potentials: npt.ArrayLike,
                       deltas: JanssonDerivativeDeltas) -> npt.ArrayLike:
                       ...

@runtime_checkable
class ConditionableComputableProperty(ComputableProperty, Protocol):
    "Can be in the denominator of a Jansson derivative"
    def jansson_deltas(self, spec, state) -> JanssonDerivativeDeltas:
        ...