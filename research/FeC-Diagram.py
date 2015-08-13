import matplotlib.pyplot as plt
from pycalphad import Database, Model, binplot
from pycalphad.core.utils import make_callable
import pycalphad.variables as v

db = Database('Fe-C_Fei_Brosh_2014_09.TDB')
fig = binplot(db, ['FE', 'C', 'VA'], list(db.phases.keys()), 'X(C)', 300.0, 2000.0, P=1e10)
plt.savefig('FeC.png')
