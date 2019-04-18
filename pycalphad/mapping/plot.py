"""
Plotting lines from ZPF boundaries in the same style as eqplot
"""

import pycalphad.variables as v
import matplotlib.pyplot as plt
from pycalphad.plot.eqplot import _axis_label


def binary_plot(zpf_boundary_sets, tielines=True, ax=None):
    """

    Parameters
    ----------
    zpf_boundary_sets : pycalphad.mapping.zpf_boundary_sets.ZPFBoundarySets
    tielines : bool

    Returns
    -------

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    scatter_dict, tieline_coll, legend_handles = zpf_boundary_sets.get_plot_boundaries()
    ax.scatter(scatter_dict['x'], scatter_dict['y'], c=scatter_dict['c'], edgecolor='None', s=3, zorder=2)
    if tielines:
        ax.add_collection(tieline_coll)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True)
    plot_title = '-'.join([component for component in sorted(zpf_boundary_sets.components) if component != 'VA'])
    ax.set_title(plot_title, fontsize=20)
    ax.set_xlabel(_axis_label(zpf_boundary_sets.indep_comp), labelpad=15, fontsize=20)
    ax.set_ylabel(_axis_label(v.T), fontsize=20)
    ax.set_xlim(0, 1)

