"""
The binary module enables plotting of binary
isobaric phase diagrams.
"""
import scipy.spatial
import itertools
import numpy as np
import matplotlib.pyplot as plt
#pylint: disable=E1101
from matplotlib import collections as mc
from pycalphad.minimize import eq

def binplot(db, comps, phases, x_variable, low_temp, high_temp, **kwargs):
    """
    Calculate the binary isobaric phase diagram for the given temperature
    range.
    """
    assert high_temp > low_temp
    minimum_distance = 0.1
    tie_lines = []
    tie_line_colors = []
    tie_line_widths = []
    steps = 0
    try:
        steps = kwargs['steps']
    except KeyError:
        steps = int((high_temp-low_temp) / 10) # Take 10 K steps by default
    temps = np.linspace(low_temp, high_temp, num=steps)

    ppp = 300 # points per phase
    if 'points_per_phase' not in kwargs:
        kwargs['points_per_phase'] = ppp

    for temp in temps:
        # Calculate energy surface at each temperature
        full_df = eq(db, comps, phases, T=temp, **kwargs)

        # Select only the P, T, etc., of interest
        point_selector = (full_df['T'] == temp)
        #for variable, value in statevars.items():
        #    point_selector = point_selector & (df[variable] == value)
        hull_frame = full_df.ix[point_selector, \
            [x_variable, 'GM', 'Phase', 'T']]
        #print(hull_frame)
        point_frame = hull_frame[[x_variable]]
        # Calculate the convex hull for the desired points
        hull = scipy.spatial.ConvexHull(
            hull_frame[[x_variable, 'GM']].values
        )
        # keep track of tie line orientations
        # this is for invariant reaction detection
        tieline_normals = []
        current_tielines = []
        for simplex, equ in zip(hull.simplices, hull.equations):
            if equ[-2] > -1e-5:
                # simplex oriented 'upwards' in energy direction
                # must not be part of the energy surface
                continue
            distances = \
                scipy.spatial.distance.pdist(
                    point_frame.iloc[simplex]
                )
            simplex_edges = \
                np.asarray(list(itertools.combinations(simplex, 2)))
            phase_edge_list = hull_frame['Phase'].values[simplex_edges]
            edge_phases_match = \
                np.array(phase_edge_list[:, 0] == phase_edge_list[:, 1])
            new_lines = \
                simplex_edges[
                    (distances >= minimum_distance) | (~edge_phases_match)
                ] # tie line because of length
            # or tie line because it connects two different phases
            if len(new_lines) == 1:
                # This is a two phase region
                first_endpoint = [point_frame.iat[new_lines[0][0], 0], temp]
                second_endpoint = [point_frame.iat[new_lines[0][1], 0], temp]
                current_tielines.append([first_endpoint, second_endpoint])
                tieline_norm = equ[:-1]/np.linalg.norm(equ[:-1])

                for idx, normal in enumerate(tieline_normals):
                    dihedral = abs(np.dot(normal, tieline_norm))
                    if dihedral > 0.99999999999:
                        # nearly coplanar: we are near a 3-phase boundary
                        # red for an invariant
                        tie_lines.append(current_tielines[-1])
                        tie_lines.append(current_tielines[idx])
                        # prevent double counting
                        del current_tielines[idx]
                        del current_tielines[-1]
                        tie_line_colors.append([1, 0, 0, 1])
                        tie_line_widths.append(2)
                        tie_line_colors.append([1, 0, 0, 1])
                        tie_line_widths.append(2)

                tieline_normals.append(tieline_norm)

                for line in current_tielines:
                    # Green for a tie line
                    tie_lines.append(line)
                    tie_line_colors.append([0, 1, 0, 1])
                    tie_line_widths.append(0.5)

            elif len(new_lines) == 0:
                # Single-phase region; drop this simplex
                pass
    tie_lines = np.asarray(tie_lines)
    #print(tie_lines)
    # Final plotting setup

    fig = plt.figure(dpi=600, figsize=(6, 6))
    ax = fig.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True)
    plt.xlim([0, 1])
    plt.ylim([low_temp, high_temp])
    if len(tie_lines) > 0:
        lc = mc.LineCollection(
            tie_lines, color=tie_line_colors, linewidth=tie_line_widths
        )
        ax.add_collection(lc)
    ax.scatter(tie_lines[:, :, 0], tie_lines[:, :, 1], color='black')


    plt.title('Diagram', fontsize=25)
    ax.set_xlabel(x_variable, labelpad=15, fontsize=20)
    ax.set_ylabel("Temperature", fontsize=20)
    plt.show()
