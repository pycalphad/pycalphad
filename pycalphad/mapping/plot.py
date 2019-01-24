"""
Plotting lines from ZPF boundaries in the same style as eqplot
"""

import pycalphad.variables as v
import matplotlib.pyplot as plt
from pycalphad.plot.eqplot import _axis_label


def binary_plot(zpf_boundary_sets, tielines=True):
    """

    Parameters
    ----------
    zpf_boundary_sets : pycalphad.mapping.zpf_boundary_sets.ZPFBoundarySets
    tielines : bool

    Returns
    -------

    """
    fig = plt.figure()
    ax = fig.gca()
    pths, tieline_tups, legend_handles = zpf_boundary_sets.get_plot_boundary_paths()
    for pth in pths:
        ax.plot(pth[0], pth[1], c=pth[2], markersize=3, zorder=2)
        # ax.scatter(pth[0], pth[1], c=pth[2], edgecolor='None', s=3, zorder=2)
        if tielines:
            for xs, ys, color in tieline_tups:
                ax.plot(xs, ys, c=color, marker='None', zorder=1, linewidth=0.5)

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

