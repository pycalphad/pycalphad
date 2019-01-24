from collections import defaultdict
from pycalphad.plot.utils import phase_legend


class ZPFBoundarySets():
    def __init__(self, ):
        self.boundaries = defaultdict(list)
        self.compset_groups = []
        self.boundary_sets = [[]]
        self.current_boundary_set = self.boundary_sets[0]


    def get_phases(self):
        """Get the phases in each boundary set

        Notes
        -----
        For this, assume the first compset has all the phases for this boundary
        set, which should be true because each boundary set should be for one
        particular region corresponding to a set of StartPoints.
        """
        phases_set = set()
        for bs in self.boundary_sets:
            if len(bs) >= 1:
                first_compset = bs[0]
                phases_set = phases_set.union({c.phase_name for c in first_compset})
        return sorted(phases_set)

    def add_boundary_set(self,):
        """
        Sets the current boundary set

        Returns
        -------

        """
        if len(self.current_boundary_set) > 0:
            self.current_boundary_set = []
            self.boundary_sets.append(self.current_boundary_set)

    def add_compsets(self, compsets):
        """
        Add composition sets to current boundary sets

        Parameters
        ----------
        compsets :

        Returns
        -------

        """
        self.current_boundary_set.append(compsets)


    def __repr__(self, ):
        phase_string = "/".join(
            ["{}: {}".format(p, len(v)) for p, v in self.boundaries.items()])
        return "ZPFBoundarySets<{}>".format(phase_string)

    def get_plot_boundary_points(self, ):
        """
        Return a list of ZPF compsets as paths to plot
        """
        raise NotImplementedError()  # needs to be re-written to manage new boundary sets format
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

    def get_plot_boundary_paths(self, ):
        """
        Return a list of ZPF compsets as paths to plot
        """
        legend_handles, colors = phase_legend(self.get_phases())
        plot_tuples = []
        tieline_tuples = []
        for boundary_set in self.boundary_sets:
            if len(boundary_set) < 1:
                continue
            x_boundary_dict = defaultdict(list)
            T_boundary_dict = defaultdict(list)
            for compsets in boundary_set:
                x_tieline = []
                T_tieline = []
                for c in compsets:
                    phase_name = c.phase_name
                    x_boundary_dict[phase_name].append(c.composition)
                    T_boundary_dict[phase_name].append(c.temperature)
                    x_tieline.append(c.composition)
                    T_tieline.append(c.temperature)
                if len(x_tieline) == 2:
                    tieline_col = [0, 1, 0, 1]
                elif len(x_tieline) == 3:
                    tieline_col = [1, 0, 0, 1]
                tieline_tuples.append((x_tieline, T_tieline, tieline_col))
            for phase_name in x_boundary_dict.keys():
                plot_tup = (x_boundary_dict[phase_name], T_boundary_dict[phase_name], colors[phase_name])
                plot_tuples.append(plot_tup)
        return plot_tuples, tieline_tuples, legend_handles


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


