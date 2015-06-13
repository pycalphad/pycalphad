import matplotlib.pyplot as plt
import numpy as np
from pycalphad import Database, Model
from pycalphad.eq.utils import make_callable
import pycalphad.variables as v

db = Database('Fe-C_Fei_Brosh_2014_09.TDB')
mod = Model(db, ['FE', 'C', 'VA'], 'FCC_A1')
from pycalphad.eq.energy_surf import energy_surf
data = energy_surf(db, ['FE', 'C', 'VA'], ['FCC_A1'], output='GM',\
                         T=np.linspace(300.0, 1200.0, num=10), P=np.logspace(5, 10, num=10), pdens=10)
data.to_csv('FeC-energies.csv')
