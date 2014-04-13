import os, sys
sys.path.append(os.getcwd())

import libpygibbs

# Load thermodynamic database
maindb = libpygibbs.Database("crfeni_mie.tdb")
if maindb == None:
	sys.exit("Failed to load database")

# Set equilibrium conditions
conds = libpygibbs.evalconditions()
conds.statevars['T'] = 300
conds.statevars['P'] = 101325
conds.statevars['N'] = 1
conds.elements.append("FE")
conds.elements.append("NI")
conds.elements.append("CR")
conds.elements.append("VA")
conds.xfrac["NI"] = .08
conds.xfrac["CR"] = .18
conds.phases["HCP_A3"] = libpygibbs.PhaseStatus.ENTERED
#conds.phases["BCC_A2"] = libpygibbs.PhaseStatus.ENTERED
#conds.phases["FCC_A1"] = libpygibbs.PhaseStatus.ENTERED
#conds.phases["LIQUID"] = libpygibbs.PhaseStatus.ENTERED
#conds.phases["SIGMA"] = libpygibbs.PhaseStatus.ENTERED

# Build the minimization engine
eqfact = libpygibbs.EquilibriumFactory()
if eqfact == None:
	sys.exit("Failed to build EquilibriumFactory")

# Construct equilibrium
myeq = eqfact.create(maindb, conds)

# Show result
print myeq


