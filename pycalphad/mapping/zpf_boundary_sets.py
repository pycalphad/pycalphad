from collections import defaultdict
from pycalphad.plot.utils import phase_legend


class ZPFBoundarySets():
    def __init__(self, ):
        self.boundaries = defaultdict(list)

    def add_compsets(self, *compsets):
        for compset in compsets:
            self.boundaries[compset.phase_name].append(compset)

    def __repr__(self, ):
        phase_string = "/".join(
            ["{}: {}".format(p, len(v)) for p, v in self.boundaries.items()])
        return "ZPFBoundarySets<{}>".format(phase_string)

    def boundary_paths(self, ):
        """
        Return a list of ZPF compsets as paths to plot

        Parameters
        ----------
        phase : string
            Phase name of the path to return
        """
        phases = sorted(list(self.boundaries.keys()))
        legend_handles, colors = phase_legend(phases)
        # for now, just prepare to scatter plot
        plot_tuples = []
        for p in phases:
            cs = self.boundaries[p]
            comps = [c.composition for c in cs]
            temps = [c.temperature for c in cs]
            tup = (comps, temps, colors[p], legend_handles,)
            plot_tuples.append(tup)
        return plot_tuples


        # TODO: algorithm
        # Predefine:
        # * maximum distance between points to be considered connected (e.g. FCC near x=0 and x=1 might not be connected)
        # 1. Make a copy of the list of boundaries
        # 2. Pick a starting point
        # 3. Find the index of the nearest neighbor in the list of remaining compsets (e.g. scipy)
        # 4. Pop it from the list and add it to the list of this boundary

    def boundary_path(self, phase):
        """
        Return a list of ZPF compsets as paths to plot

        Parameters
        ----------
        phase : string
            Phase name of the path to return
        """
        phases = sorted(list(self.boundaries.keys()))
        legend_handles, colors = phase_legend(phases)
        # for now, just prepare to scatter plot
        plot_tuples = []
        for p in phases:
            cs = self.boundaries[p]
            comps = [c.composition for c in cs]
            temps = [c.temperature for c in cs]
            tup = (comps, temps, colors[p], legend_handles,)
            plot_tuples.append(tup)
        return plot_tuples


        # TODO: algorithm
        # Predefine:
        # * maximum distance between points to be considered connected (e.g. FCC near x=0 and x=1 might not be connected)
        # 1. Make a copy of the list of boundaries
        # 2. Pick a starting point
        # 3. Find the index of the nearest neighbor in the list of remaining compsets (e.g. scipy)
        # 4. Pop it from the list and add it to the list of this boundary
