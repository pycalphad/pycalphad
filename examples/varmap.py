import calphad.libcalphadcpp as lcp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class GlobMin(lcp.GlobalMinimizer):
   def triangulate_hull(self):
       raw_entries = lcp.GlobalMinimizer.get_hull_entries(self)
       facets = lcp.GlobalMinimizer.get_facets(self)
       entries = []
       facet_list = []
       point_phases = []
       for entry in raw_entries:
	       stripped_entry = []
	       for coords in entry.global_coordinates:
		       stripped_entry.append(coords.data())
	       #stripped_entry.append(entry.energy)
	       point_phases.append(entry.phase_name)
	       entries.append(stripped_entry)
       for facet in facets:
	       facet_list.append(facet.vertices)
       return entries, facet_list, point_phases


# Load thermodynamic database
maindb = lcp.Database("crfeni_mie.tdb")
if maindb == None:
	sys.exit("Failed to load database")

# Set equilibrium conditions
conds = lcp.evalconditions()
conds.statevars['T'] = 2500
conds.statevars['P'] = 101325
conds.statevars['N'] = 1
conds.elements.append("FE")
conds.elements.append("NI")
conds.elements.append("CR")
conds.elements.append("VA")
conds.xfrac["NI"] = .08
conds.xfrac["CR"] = .18
conds.phases["HCP_A3"] = lcp.PhaseStatus.SUSPENDED
conds.phases["BCC_A2"] = lcp.PhaseStatus.SUSPENDED
conds.phases["FCC_A1"] = lcp.PhaseStatus.SUSPENDED
conds.phases["LIQUID"] = lcp.PhaseStatus.ENTERED
conds.phases["SIGMA"] = lcp.PhaseStatus.SUSPENDED

indices = lcp.IndexBiMap()
varmap = lcp.build_variable_map ( maindb.get_phases(), conds, indices )
cmps = {}

for ph in maindb.get_phases():
    status = conds.phases[ph.key()]
    if (status == lcp.PhaseStatus.ENTERED):
        cmps[ph.key()] = lcp.CompositionSet(ph.data(), maindb.get_parameter_set(), varmap, indices)
print ("Starting global minimization")
globminengine = GlobMin(cmps, varmap, conds)

print("Getting hull")
xy, triangles, point_phases = globminengine.triangulate_hull()
xy = np.asarray(xy)
x = xy[:,0]
y = xy[:,1]
z = 1 - x - y

new_x = 0.5*(2*y + z)/(x+y+z)
new_y = (np.sqrt(3)/2)*z/(x+y+z)
triangles = np.asarray(triangles)
colorlist = {}
import colorsys
N = len(conds.phases)
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

phasecount = 0
for phase in conds.phases:
	colorlist[phase.key()] = RGB_tuples[phasecount]
	phasecount = phasecount + 1

plotcolors = map(lambda x: colorlist[x], point_phases)

colormap = LinearSegmentedColormap('PhaseColor', plotcolors)

plt.figure(dpi=600)
plt.xlim([-0.01,1])
plt.ylim([-0.01,1])
plt.gca().set_aspect('equal')
plt.triplot(new_x, new_y, triangles, 'go-', marker=None, zorder=1)
plt.scatter(new_x, new_y, cmap=colormap, marker='.', zorder=2)
plt.title('triplot of user-specified triangulation')
plt.xlabel('X(A)')
plt.ylabel('X(B)')
plt.show()