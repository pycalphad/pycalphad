# This unfortunate monkey patch is necessary to make Py27, Py33 and Py34 work
# Source: http://stackoverflow.com/questions/34124270/pickling-method-descriptor-objects-in-python

# first import dill, which populates itself into pickle's dispatch
import dill
import pickle
# save the MethodDescriptorType from dill
MethodDescriptorType = type(type.__dict__['mro'])
if pickle.__dict__.get('_Pickler', None):
    MethodDescriptorWrapper = pickle._Pickler.dispatch[MethodDescriptorType]
else:
    MethodDescriptorWrapper = pickle.Pickler.dispatch[MethodDescriptorType]
# cloudpickle does the same, so let it update the dispatch table
import cloudpickle
# now, put the saved MethodDescriptorType back in
if pickle.__dict__.get('_Pickler', None):
    pickle._Pickler.dispatch[MethodDescriptorType] = MethodDescriptorWrapper
else:
    pickle.Pickler.dispatch[MethodDescriptorType] = MethodDescriptorWrapper

import warnings
warnings.filterwarnings('ignore', message='divide by zero encountered in log')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

import sympy.functions.elementary.piecewise
import pycalphad.core.patched_piecewise
sympy.functions.elementary.piecewise.Piecewise.eval = classmethod(pycalphad.core.patched_piecewise.piecewise_eval)

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
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
