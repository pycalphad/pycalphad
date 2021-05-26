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

# Set the version of pycalphad
try:
    from ._dev import get_version
    # We have a local (editable) installation and can get the version based on the
    # source control management system at the project root.
    __version__ = get_version(root='..', relative_to=__file__)
    del get_version
except ImportError:
    # Fall back on the metadata of the installed package
    try:
        from importlib.metadata import version
    except ImportError:
        # backport for Python<3.8
        from importlib_metadata import version
    __version__ = version("pycalphad")
    del version
