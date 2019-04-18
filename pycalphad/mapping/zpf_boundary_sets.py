from collections import defaultdict
from pycalphad.plot.utils import phase_legend
from pycalphad.mapping.compsets import CompSet2D
from pycalphad.mapping.utils import sort_x_by_y
from matplotlib.collections import LineCollection
import numpy as np


class TwoPhaseRegion():
    """
    A group of points that belong to the same phase region.

    Attributes
    ----------
    phases : frozenset
        Unique phases in this two phase region
    compsets : list of CompSet2D
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
        compsets : CompSet2D
        Xtol : float
        Ttol : float

        Returns
        -------
        bool

        """
        if compsets.unique_phases == self.phases:
            last_compsets = self.compsets[-1]
            if np.all(last_compsets.pairwise_xdiscrepancy(compsets) < Xtol) and \
               np.abs(last_compsets.temperature - compsets.temperature) < Ttol:
                return True
        return False

    def add_compsets(self, compsets):
        """

        Parameters
        ----------
        compsets : CompSet2D

        Notes
        -----
        Users are responsible for determining if the compsets belong in this TwoPhaseRegion

        """
        self.compsets.append(compsets)

class ZPFBoundarySets():
    """

    Attributes
    ----------
    two_phase_regions : list of TwoPhaseRegion
    """
    def __init__(self, components, independent_component_statevar):
        self.boundaries = defaultdict(list)
        self.compset_groups = []
        self.two_phase_regions = []
        self.components = components
        self.indep_comp = independent_component_statevar

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
        compsets : CompSet2D

        Returns
        -------

        """
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

    def get_plot_boundaries(self, ):
        """
        Get the ZPF boundaries to plot from each two phase region.


        Notes
        -----
        For now, we will not support connecting regions with lines, so this
        function returns a tuple of scatter_dict and a tineline_collection.

        Examples
        --------
        >>> scatter_dict, tieline_collection, legend_handles = zpf_boundary_sets.get_plot_boundaries()  # doctest: +SKIP
        >>> ax.scatter(**scatter_dict)  # doctest: +SKIP
        >>> ax.add_collection(tieline_collection)  # doctest: +SKIP
        >>> ax.legend(handles=legend_handles)  # doctest: +SKIP

        Returns
        -------
        (scatter_dict, tieline_collection, legend_handles)
        """
        all_phases = self.get_phases()
        legend_handles, colors = phase_legend(all_phases)
        scatter_dict = {'x': [], 'y': [], 'c': []}

        tieline_segments = []
        tieline_colors = []
        for tpr in self.two_phase_regions:
            for cs in tpr.compsets:
                # TODO: X=compmosition, Y=Temperature assumption
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
                tieline_colors.append([0, 1, 0, 1])

        tieline_collection = LineCollection(tieline_segments, zorder=1, linewidths=0.5, colors=tieline_colors)
        return scatter_dict, tieline_collection, legend_handles
