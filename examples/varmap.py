import calphad.libcalphadcpp as lcp

# Load thermodynamic database
maindb = lcp.Database("crfeni_mie.tdb")
if maindb == None:
	sys.exit("Failed to load database")

# Set equilibrium conditions
conds = lcp.evalconditions()
conds.statevars['T'] = 300
conds.statevars['P'] = 101325
conds.statevars['N'] = 1
conds.elements.append("FE")
conds.elements.append("NI")
conds.elements.append("CR")
conds.elements.append("VA")
conds.xfrac["NI"] = .08
conds.xfrac["CR"] = .18
#conds.phases["HCP_A3"] = lcp.PhaseStatus.ENTERED
#conds.phases["BCC_A2"] = lcp.PhaseStatus.ENTERED
#conds.phases["FCC_A1"] = lcp.PhaseStatus.ENTERED
conds.phases["LIQUID"] = lcp.PhaseStatus.ENTERED
#conds.phases["SIGMA"] = lcp.PhaseStatus.ENTERED

indices = lcp.IndexBiMap()
varmap = lcp.build_variable_map ( maindb.get_phases(), conds, indices )
cmps = []

for ph in maindb.get_phases():
    cmp = lcp.CompositionSet(ph.data(), maindb.get_parameter_set(), varmap, indices)
    cmps.append(cmp)

for cmp in cmps:
    print cmp.name()
