"""
The plot utils module contains some useful routines related to plotting.
"""

import matplotlib.patches as mpatches
import numpy as np

def phase_legend(phases):
    """
    Build matplotlib handles for the plot legend.

    Parameters
    ----------
    phases : list
        Names of the phases.

    Returns
    -------
    A tuple containing:
    (1) A list of matplotlib handle objects
    (2) A dict mapping phase names to their RGB color on the plot

    Examples
    --------
    >>> legend_handles, colors = phase_legend(['FCC_A1', 'BCC_A2', 'LIQUID'])
    """
    colorlist = {}
    # colors from Junwei Huang, March 21 2013
    # exclude green and red because of their special meaning on the diagram
    colorvalues = ["0000FF", "FFFF00", "FF00FF", "00FFFF", "000000", "800000",
                   "008000", "000080", "808000", "800080", "008080", "808080",
                   "C00000", "00C000", "0000C0", "C0C000", "C000C0", "00C0C0",
                   "C0C0C0", "400000", "004000", "000040", "404000", "400040",
                   "004040", "404040", "200000", "002000", "000020", "202000",
                   "200020", "002020", "202020", "600000", "006000", "000060",
                   "606000", "600060", "006060", "606060", "A00000", "00A000",
                   "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0", "E00000",
                   "00E000", "0000E0", "E0E000", "E000E0", "00E0E0", "E0E0E0"]
    mxx = len(colorvalues)
    phasecount = 0
    legend_handles = []
    for phase in phases:
        phase = phase.upper()
        colorlist[phase] = "#"+colorvalues[np.mod(phasecount, mxx)]
        legend_handles.append(mpatches.Patch(color=colorlist[phase],
                                             label=phase))
        phasecount = phasecount + 1
    return legend_handles, colorlist
