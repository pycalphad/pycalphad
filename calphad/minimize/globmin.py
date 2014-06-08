"""The global minimization module provides support for calculating energy
surfaces associated with determining the global energy minimum. This is
necessary for reliable automatic miscibility gap detection.

"""
import calphad.libcalphadcpp as lcp


class GlobalMinimizer(lcp.GlobalMinimizer):
	"""
	Construct the minimum energy manifold for the given conditions.

	Attributes
	----------
	type : string
		Format of the source data.
		
	Methods
	-------
	triangulate_hull()
		Get points and facets of the global convex hull.
		
	"""
	def __init__(self, phase_dict, sublset, conditions, **kwargs):
		# Initialize minimizer
		lcp.GlobalMinimizer.__init__(self)
		allowed_opts = ('critical_edge_length',
	                   'initial_subdivisions_per_axis',
	                   'refinement_subdivisions_per_axis',
	                   'max_search_depth'
	                   )
		# Set options
		for k, v in kwargs.iteritems():
			assert( k in allowed_opts )
			setattr(self, k, v)
		
		# Execute calculation
		lcp.GlobalMinimizer.run(self, phase_dict, sublset, conditions)
	def triangulate_hull(self):
		raw_entries = lcp.GlobalMinimizer.get_hull_entries(self)
		facets = lcp.GlobalMinimizer.get_facets(self)
		entries = []
		facet_list = []
		point_phases = []
		hull_mask = []
		for entry in raw_entries:
			stripped_entry = []
			for coords in entry.global_coordinates:
				stripped_entry.append(coords.data())
			#stripped_entry.append(entry.energy)
			hull_mask.append(not entry.on_global_hull)
			point_phases.append(entry.phase_name)
			entries.append(stripped_entry)
		for facet in facets:
			facet_list.append(facet.vertices)
		return entries, facet_list, point_phases, hull_mask