import os, sys
sys.path.append(os.getcwd())

import libpygibbs

# Load thermodynamic database
maindb = libpygibbs.Database("idealbin.tdb")
if maindb == None:
	sys.exit("Failed to load database")

# Set equilibrium conditions
conds = libpygibbs.evalconditions()
conds.statevars['T'] = 1500
conds.statevars['P'] = 101325
conds.statevars['N'] = 1
conds.elements.append("NB")
conds.elements.append("RE")
conds.elements.append("VA")
conds.xfrac["NB"] = 0.6
conds.phases["CHI"] = libpygibbs.PhaseStatus.ENTERED

# Build the minimization engine
eqfact = libpygibbs.EquilibriumFactory()
if eqfact == None:
	sys.exit("Failed to build EquilibriumFactory")

# Construct equilibrium
myeq = eqfact.create(maindb, conds)

# Show result
print myeq


