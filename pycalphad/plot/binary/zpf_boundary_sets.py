from pycalphad.plot.utils import phase_legend
from pycalphad.plot.binary.compsets import CompsetPair
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
import numpy as np


class TwoPhaseRegion():
    """
    A group of points that belong to the same phase region.

    Attributes
    ----------
    phases : frozenset
        Unique phases in this two phase region
    compsets : list of CompsetPair
    """
    def __init__(self, initial_compsets):
        self.phases = initial_compsets.unique_phases
        self.compsets = [initial_compsets]

    def __repr__(self):
        phases_pprint = ", ".join(self.phases)
        s = "<TwoPhaseRegion(Phases=({}), Points={})>".format(phases_pprint, len(self.compsets))
        return s

    def compsets_belong_in_region(self, compsets, Xtol=0.05, Ttol=10):
        """
        Return True if the composition sets belong to this TwoPhaseRegion

        Parameters
        ----------
        compsets : CompsetPair
        Xtol : float
            Composition discrepancy tolerance
        Ttol : float
            Temperature discrepancy tolerance

        Returns
        -------
        bool

        Notes
        -----
        A new CompsetPair object belongs in the this region if, when compared to the most recently added CompsetPair:

        1. All the phases are the same (in the same order)
        2. The composition discrepancies between BinaryCompsets of the same phase are below Xtol
        3. The temperature discrepancy between the CompsetPairs is below Ttol

        """
        if compsets.unique_phases == self.phases:
            last_compsets = self.compsets[-1]
            if np.all(last_compsets.pairwise_xdiscrepancy(compsets) < Xtol) and \
               np.abs(last_compsets.temperature - compsets.temperature) < Ttol:
                return True
        return False

    def add_compsets(self, compset_pair):
        """
        Add composition sets to this region.

        Parameters
        ----------
        pair : CompsetPair
            Add a CompsetPair to the list of composition sets

        Notes
        -----
        For performance reasons (checking whether the CompsetPair belongs has a
        complexity of O(N)), users are responsible for determining if the
        compsets belong in this TwoPhaseRegion.

        """
        self.compsets.append(compset_pair)


class ZPFBoundarySets():
    """
    Holds CompsetPairs that can be organized into TwoPhaseRegions and plotted
    together.

    Attributes
    ----------
    components : list of str
        List of components
    indep_comp_cond : v.X
        Condition for the independent component
    all_compsets : list of CompsetPair
    two_phase_regions : list of TwoPhaseRegion

    """
    def __init__(self, comps, indep_composition_condition):
        self.components = comps
        self.indep_comp_cond = indep_composition_condition
        self.all_compsets = []
        self.two_phase_regions = []

    def get_phases(self):
        """
        Get all the phases represented in the two phase regions
        """
        phases_set = set().union(*[tpr.phases for tpr in self.two_phase_regions])
        return sorted(phases_set)

    def add_compsets(self, compsets, Xtol=0.05, Ttol=10):
        """
        Add composition sets to current boundary sets

        Parameters
        ----------
        compsets : CompsetPair

        """
        self.all_compsets.append(compsets)
        if len(self.two_phase_regions) == 0:
            self.two_phase_regions.append(TwoPhaseRegion(compsets))
        else:
            for tpr in self.two_phase_regions:
                if tpr.compsets_belong_in_region(compsets, Xtol=Xtol, Ttol=Ttol):
                    tpr.add_compsets(compsets)
                    break
            else:
                self.two_phase_regions.append(TwoPhaseRegion(compsets))

    def __repr__(self, ):
        phase_string = "/".join(
            ["{}: {}".format(p, len(v)) for p, v in self.boundaries.items()])
        return "ZPFBoundarySets<{}>".format(phase_string)

    def rebuild_two_phase_regions(self, Xtol=0.05, Ttol=10):
        """
        Rebuild the two phase regions with new tolerances.

        Parameters
        ----------
        Xtol : float
            See TwoPhaseRegion.compsets_belong_in_region
        Ttol : float
            See TwoPhaseRegion.compsets_belong_in_region

        """
        self.two_phase_regions = []
        previous_all_compsets = self.all_compsets
        self.all_compsets = []
        for cs in previous_all_compsets:
            self.add_compsets(cs, Xtol=Xtol, Ttol=Ttol)

    def get_scatter_plot_boundaries(self, tieline_color=(0, 1, 0, 1), legend_generator=phase_legend):
        """
        Get the ZPF boundaries to plot from each two phase region.

        Parameters
        ----------
        tieline_color: color
            A valid matplotlib color, such as a named color string, hex RGB
            string, or a tuple of RGBA components to set the color of the two
            phase region tielines. The default is an RGBA tuple for green:
            (0, 1, 0, 1).
        legend_generator : Callable
            A function that will be called with the list of phases and will
            return legend labels and colors for each phase. By default
            pycalphad.plot.utils.phase_legend is used

        Notes
        -----
        For now, we will not support connecting regions with lines, so this
        function returns a tuple of scatter_dict and a tineline_collection.

        Examples
        --------
        >>> scatter_dict, tieline_collection, legend_handles = zpf_boundary_sets.get_scatter_plot_boundaries()  # doctest: +SKIP
        >>> ax.scatter(**scatter_dict)  # doctest: +SKIP
        >>> ax.add_collection(tieline_collection)  # doctest: +SKIP
        >>> ax.legend(handles=legend_handles)  # doctest: +SKIP

        Returns
        -------
        (scatter_dict, tieline_collection, legend_handles)
        """
        all_phases = self.get_phases()
        legend_handles, colors = legend_generator(all_phases)
        scatter_dict = {'x': [], 'y': [], 'c': []}

        tieline_segments = []
        tieline_colors = []
        for tpr in self.two_phase_regions:
            for cs in tpr.compsets:
                # TODO: X=composition, Y=Temperature assumption
                xs = cs.compositions.tolist()
                ys = [cs.temperature, cs.temperature]
                phases = cs.phases

                # prepare scatter dict
                scatter_dict['x'].extend(xs)
                scatter_dict['y'].extend(ys)
                scatter_dict['c'].extend([colors[p] for p in phases])

                # add tieline segment segment
                # a segment is a list of [[x1, y1], [x2, y2]]
                tieline_segments.append(np.array([xs, ys]).T)
                # always a two phase region, green lines
                tieline_colors.append(tieline_color)

        tieline_collection = LineCollection(tieline_segments, zorder=1, linewidths=0.5, colors=tieline_colors)
        return scatter_dict, tieline_collection, legend_handles

    def get_line_plot_boundaries(self, close_miscibility_gaps=0.05, tieline_color=(0, 1, 0, 1), legend_generator=phase_legend):
        """
        Get the ZPF boundaries to plot from each two phase region.

        Parameters
        ----------
        close_miscibility_gaps : float, optional
            If a float is passed, add a line segment between compsets at the top
             or bottom of a two phase region if the discrepancy is below a
             tolerance. If `None` is passed, do not close the gap.
        tieline_color: color
            A valid matplotlib color, such as a named color string, hex RGB
            string, or a tuple of RGBA components to set the color of the two
            phase region tielines. The default is an RGBA tuple for green:
            (0, 1, 0, 1).
        legend_generator : Callable
            A function that will be called with the list of phases and will
            return legend labels and colors for each phase. By default
            pycalphad.plot.utils.phase_legend is used

        Notes
        -----
        For now, we will not support connecting regions with lines, so this
        function returns a tuple of scatter_dict and a tineline_collection.

        Examples
        --------
        >>> boundary_collection, tieline_collection, legend_handles = zpf_boundary_sets.get_line_plot_boundaries()  # doctest: +SKIP
        >>> ax.add_collection(boundary_collection)  # doctest: +SKIP
        >>> ax.add_collection(tieline_collection)  # doctest: +SKIP
        >>> ax.legend(handles=legend_handles)  # doctest: +SKIP

        Returns
        -------
        (line_collections, tieline_collection, legend_handles)
        """
        # TODO: add some tracking of the endpoints/startpoints and join them with
        #       a new line segment if they are close.
        all_phases = self.get_phases()
        legend_handles, colors = legend_generator(all_phases)
        tieline_segments = []
        tieline_colors = []
        boundary_segments = []
        boundary_colors = []
        for tpr in self.two_phase_regions:
            # each two phase region contributes two line collections, one for
            # each ZPF line
            a_path = []
            b_path = []
            for cs in tpr.compsets:
                # TODO: X=composition, Y=Temperature assumption
                xs = cs.compositions.tolist()
                ys = [cs.temperature, cs.temperature]
                a_path.append([xs[0], ys[0]])
                b_path.append([xs[1], ys[1]])
                # add tieline segment segment
                # a segment is a list of [[x1, y1], [x2, y2]]
                tieline_segments.append(np.array([xs, ys]).T)
                # always a two phase region, green lines
                tieline_colors.append(tieline_color)

            # build the line collections for each two phase region
            ordered_phases = tpr.compsets[0].phases
            a_color = to_rgba(colors[ordered_phases[0]])
            b_color = to_rgba(colors[ordered_phases[1]])

            # close miscibility gaps, both top and bottom
            if close_miscibility_gaps is not None and len(tpr.phases) == 1:
                bottom_cs = tpr.compsets[0]
                if bottom_cs.xdiscrepancy() < close_miscibility_gaps:
                    boundary_segments.append(np.array([bottom_cs.compositions, [bottom_cs.temperature, bottom_cs.temperature]]).T)
                    boundary_colors.append(colors[ordered_phases[0]])  # colors are the same
                top_cs = tpr.compsets[-1]
                if top_cs.xdiscrepancy() < close_miscibility_gaps:
                    boundary_segments.append(np.array([top_cs.compositions, [top_cs.temperature, top_cs.temperature]]).T)
                    boundary_colors.append(colors[ordered_phases[0]])  # colors are the same

            boundary_segments.append(a_path)
            boundary_segments.append(b_path)
            boundary_colors.append(a_color)
            boundary_colors.append(b_color)
        boundary_collection = LineCollection(boundary_segments, colors=boundary_colors)
        tieline_collection = LineCollection(tieline_segments, zorder=1, linewidths=0.5, colors=tieline_colors)
        return boundary_collection, tieline_collection, legend_handles
