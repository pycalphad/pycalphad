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
		"""
		Construct the convex hull to find the minimum energy surface.
		
		Parameters
		----------
		phase_dict : dict
		    Dictionary of CompositionSet objects.
		sublset : sublattice_set
		    Sublattice configuration of all phases.
		conditions : evalconditions
		critical_edge_length : double, optional
		    Minimum length of a tie line. (default is 0.005)
		initial_subdivisions_per_axis : int, optional
		    Number of axis subdivisions along each axis. (default is 20)
		refinement_subdivisions_per_axis : int, optional
		    Number of axis subdivisions during mesh refinement. (default is 2)
		max_search_depth : int, optional
		    Maximum recursive depth for mesh refinement. (default is 5)
		    
		
		Examples
		--------
		TODO: None yet.
		"""
		# Initialize minimizer on C++ side
		lcp.GlobalMinimizer.__init__(self)
		# Create list of allowed options
		allowed_opts = ('critical_edge_length',
	                   'initial_subdivisions_per_axis',
	                   'refinement_subdivisions_per_axis',
	                   'max_search_depth'
	                   )
		# Set options; these will override C++ settings
		for k, v in kwargs.iteritems():
			assert( k in allowed_opts )
			setattr(self, k, v)
		
		# Execute calculation on C++ side
		lcp.GlobalMinimizer.run(self, phase_dict, sublset, conditions)
	def triangulate_hull(self):
		"""
		Get the triangulation of the convex hull.
		
		Parameters
		----------
		None.
		
		Returns
		-------
		entries : ordered list of points on the convex hull for all phases
		facet_list : ordered list of vertices comprising each facet
		point_phases : ordered list of the phase where each point comes from
		hull_mask : ordered list of bools; true if _not_ on global hull
		    
		See Also
		--------
		Nothing.

		"""
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