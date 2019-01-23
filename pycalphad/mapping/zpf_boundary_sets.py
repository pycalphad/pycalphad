from collections import defaultdict
from pycalphad.plot.utils import phase_legend


class ZPFBoundarySets():
    def __init__(self, ):
        self.boundaries = defaultdict(list)
        self.compset_groups = []

    def add_compsets(self, *compsets):
        self.compset_groups.append(compsets)
        for compset in compsets:
            self.boundaries[compset.phase_name].append(compset)

    def __repr__(self, ):
        phase_string = "/".join(
            ["{}: {}".format(p, len(v)) for p, v in self.boundaries.items()])
        return "ZPFBoundarySets<{}>".format(phase_string)

    def get_plot_boundary_paths(self, ):
        """
        Return a list of ZPF compsets as paths to plot
        """
        phases = sorted(list(self.boundaries.keys()))
        legend_handles, colors = phase_legend(phases)
        # for now, just prepare to scatter plot
        plot_tuples = []
        for p in phases:
            cs = self.boundaries[p]
            comps = [c.composition for c in cs]
            temps = [c.temperature for c in cs]
            tup = (comps, temps, colors[p],)
            plot_tuples.append(tup)
        return plot_tuples, legend_handles

    def get_plot_tielines(self):
        """
        Get paths for tielines to plot.

        Notes
        -----
        These are just the paths for the lines. They should be plotted without
        markers and with boundary paths to be meaningful.

        Returns
        -------
        List of tuples of (xs, ys, colors) of tielines to plot
        """
        tieline_plot_tuples = []
        for compset_group in self.compset_groups:
            xs = [c.composition for c in compset_group]
            ys = [c.temperature for c in compset_group]
            if len(compset_group) == 2:
                c = [0, 1, 0, 1]
            elif len(compset_group) == 3:
                c = [1, 0, 0, 1]
            tieline_plot_tuples.append((xs, ys, c,))
        return tieline_plot_tuples


