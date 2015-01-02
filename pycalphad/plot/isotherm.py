"""
The isotherm module enables plotting of ternary
isobaric-isothermal phase diagrams.
"""
import scipy.spatial
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import pycalphad.plot.projections.triangular #pylint: disable=W0611

def isotherm(df, x_variable, y_variable, **statevars):
    """
    Calculate the isothermal-isobaric phase diagram at the given temperature.
    """
    # Select only the T, P, etc., of interest
    point_selector = np.ones(len(df), dtype=bool)
    for variable, value in statevars.items():
        point_selector = point_selector & (df[variable] == value)

    hull_frame = df.ix[point_selector, [x_variable, y_variable, 'GM', 'Phase']]
    #print(hull_frame)
    point_frame = hull_frame[[x_variable, y_variable]]
    # Calculate the convex hull for the desired points
    hull = scipy.spatial.ConvexHull(
        hull_frame[[x_variable, y_variable, 'GM']].values
    )
    point_mask = np.ones(len(point_frame.index), dtype=bool) # mask all points

    # unmask any point that is an endpoint for a tieline
    minimum_distance = 0.05
    tie_lines = []
    tie_line_colors = []
    tie_line_widths = []
    for simplex, equ in zip(hull.simplices, hull.equations):
        if equ[-2] > -1e-5:
            # simplex oriented 'upwards' in energy direction
            # must not be part of the energy surface
            continue
        #if np.any(np.any(
        #        np.where(
        #            point_frame.iloc[simplex] < 1e-3,\
        #            True, False), axis=1
        #    ), axis=0):
        #    # this simplex is not part of the energy surface
        #    continue
        #if np.all(np.any(
        #        np.where(
        #            point_frame.iloc[simplex] > 0.98,\
        #            True, False), axis=1
        #    ), axis=0):
        #    # this simplex is not part of the energy surface
        #    continue
        distances = \
            scipy.spatial.distance.pdist(
                point_frame.iloc[simplex]
            )
        shortest_distance = np.min(distances)
        simplex_edges = np.asarray(list(itertools.combinations(simplex, 2)))
        phase_edge_list = hull_frame['Phase'].values[simplex_edges]
        edge_phases_match = \
            np.array(phase_edge_list[:, 0] == phase_edge_list[:, 1])
        new_lines = \
            simplex_edges[
                (distances >= minimum_distance) | (~edge_phases_match)
            ] # tie line because of length
        # or tie line because it connects two different phases
        if len(new_lines) == 3:
            # This is a three phase region because
            # all simplices satisfy the minimum distance requirement
            tie_lines.extend(new_lines)
            point_mask[new_lines] = False
            # Red for a tie plane
            tie_line_colors.append([1, 0, 0, 1])
            tie_line_colors.append([1, 0, 0, 1])
            tie_line_colors.append([1, 0, 0, 1])
            tie_line_widths.append(3)
            tie_line_widths.append(3)
            tie_line_widths.append(3)
        elif len(new_lines) == 2:
            # This is a two phase region
            shortest_edge = \
                simplex_edges[np.where(distances == shortest_distance)][0]
            other_tiepoint_index = \
                np.setdiff1d(simplex, shortest_edge, assume_unique=True)[0]
            # For the short edge, we need to choose the vertex with
            # the minimum energy.
            # It is more likely to be on the true tie-line since nearby
            # metastable points will be higher in energy.
            minimum_energy_vertex = \
                np.argmin(hull_frame['GM'].iloc[shortest_edge])
            tie_lines.append([other_tiepoint_index, minimum_energy_vertex])
            point_mask[[other_tiepoint_index, minimum_energy_vertex]] = False
            # Green for a tie line
            tie_line_colors.append([0, 1, 0, 1])
            tie_line_widths.append(0.5)
        elif len(new_lines) == 0:
            # Single-phase region; drop this simplex
            pass
    tie_lines = np.asarray(tie_lines)
    # Final plotting setup
    
    # Mask the metastable points
    masked_x = np.ma.array(hull_frame[x_variable], mask=point_mask)
    masked_y = np.ma.array(hull_frame[y_variable], mask=point_mask)
    
    fig = plt.figure(dpi=600,figsize=(12,12))
    ax = fig.gca(projection="triangular") # use ternary axes
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True)
    plt.xlim([-0.01,1])
    plt.ylim([-0.01,1])
    plt.gca().set_aspect('equal')
    if len(tie_lines) > 0:
        lc = mc.LineCollection(
            hull_frame[[x_variable,y_variable]].values[tie_lines], \
                color=tie_line_colors, linewidth=tie_line_widths
        )
        ax.add_collection(lc)
    ax.scatter(masked_x, masked_y, color='black')
    
    ax.text(0.3, 0.8, 'T = '+str(statevars['T'])+ ' K',
            verticalalignment='bottom', horizontalalignment='left',
            color='black', fontsize=20)
    
    
    plt.title('Diagram',fontsize=25, x=0.5, y=0.9)
    ax.set_xlabel(x_variable, labelpad=15,fontsize=20)
    ax.set_ylabel(y_variable,rotation=60,fontsize=20,labelpad=-120)
    plt.show()
