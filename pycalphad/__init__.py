import pycalphad.variables as v
from pycalphad.model import Model
from pycalphad.io.database import Database

# Trigger format extension hooks
import pycalphad.io.tdb

from pycalphad.core.calculate import calculate
from pycalphad.core.equilibrium import equilibrium
from pycalphad.core.equilibrium import EquilibriumError, ConditionError
from pycalphad.plot.binary import binplot
from pycalphad.plot.eqplot import eqplot
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
