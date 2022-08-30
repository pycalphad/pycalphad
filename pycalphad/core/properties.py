from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
from typing import List, Optional, Any

@dataclass
class DotDerivativeDeltas:
    delta_chemical_potentials: Optional[Any]
    delta_statevars: Optional[Any]
    delta_parameters: Optional[Any]
    delta_phase_amounts: Optional[Any]
    delta_sitefracs: Optional[Any]