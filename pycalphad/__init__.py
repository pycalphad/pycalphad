import warnings
warnings.filterwarnings('ignore', message='divide by zero encountered in log')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

import sympy.functions.elementary.piecewise
import pycalphad.core.patched_piecewise
sympy.functions.elementary.piecewise.Piecewise.eval = classmethod(pycalphad.core.patched_piecewise.piecewise_eval)
sympy.functions.elementary.piecewise.ExprCondPair.__new__ = staticmethod(pycalphad.core.patched_piecewise.exprcondpair_new)

from pycalphad.core.errors import *
import pycalphad.variables as v
from pycalphad.model import Model, ReferenceState
from pycalphad.io.database import Database

# Trigger format extension hooks
import pycalphad.io.tdb

from pycalphad.core.calculate import calculate
from pycalphad.core.equilibrium import equilibrium
from pycalphad.plot.binary import binplot
from pycalphad.plot.ternary import ternplot
from pycalphad.plot.eqplot import eqplot

from setuptools_scm import get_version
__version__ = get_version()
