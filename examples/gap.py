import os, sys
sys.path.append(os.getcwd())

import libpygibbs

# Load thermodynamic database
maindb = libpygibbs.Database("alzn_mey.tdb")
if maindb == None:
	sys.exit("Failed to load database")

# Set equilibrium conditions
conds = libpygibbs.evalconditions()
conds.statevars['T'] = 340
conds.statevars['P'] = 101325
conds.statevars['N'] = 1
conds.elements.append("AL")
conds.elements.append("ZN")
conds.xfrac["AL"] = .50
#conds.phases["HCP_A3"] = libpygibbs.PhaseStatus.ENTERED
conds.phases["FCC_A1"] = libpygibbs.PhaseStatus.ENTERED

# Build the minimization engine
eqfact = libpygibbs.EquilibriumFactory()
if eqfact == None:
	sys.exit("Failed to build EquilibriumFactory")

# Construct equilibrium
myeq = eqfact.create(maindb, conds)

# Show result
print myeq


