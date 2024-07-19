"""
Helper functions to grab data from tieline strategies (BinaryStrategy and TernaryStrategy)
This is to avoid repeating get_XXX_data functions for the two strategies

TODO: Once BinaryStrategy and TernaryStrategy are merged into a more general TielineStrategy,
      these two functions can be moved into TieLineStrategy and strategy_utils.py as be removed
"""

from typing import Union

from pycalphad import variables as v

from pycalphad.mapping.primitives import _get_phase_specific_variable

def get_invariant_data_from_tieline_strategy(strategy, x: v.StateVariable, y: v.StateVariable):
    """
    Creates dictionary of data for node plotting in binary and ternary plots

    Parameters
    ----------
    strategy : BinaryStrategy or TernaryStrategy
    x : v.StateVariable
    y : v.StateVariable

    Returns
    -------
    node_data : list[dict]
        Each dict will be
        {
            "phases" : [str]
            "x" : [float]
            "y" : [float]
        }
        Indices in x and y will match the indices in phases
    """
    invariant_data = []
    for node in strategy.node_queue.nodes:
        # Nodes in binary and ternary mappings are always 3 composition sets
        if len(node.stable_composition_sets) == 3:
            node_phases = node.stable_phases_with_multiplicity
            x_data = [node.get_property(_get_phase_specific_variable(p, x)) for p in node_phases]
            y_data = [node.get_property(_get_phase_specific_variable(p, y)) for p in node_phases]
            data = {
                'phases': node_phases,
                'x': x_data,
                'y': y_data,
            }
            invariant_data.append(data)

    return invariant_data

def get_tieline_data_from_tieline_strategy(strategy, x: v.StateVariable, y: v.StateVariable):
    """
    Creates dictionary of data for plotting zpf lines

    Parameters
    ----------
    strategy : BinaryStrategy or TernaryStrategy
    x : v.StateVariable
    y : v.StateVariable

    Returns
    -------
    zpf_data : list[dict]
        Each dict will be
        {
            <phase_name> : {
                "x": [float]
                "y": [float]
            }
        }
        Length of x and y for each phase in a ZPFLine should be equal
    """
    zpf_data = []
    for zpf_line in strategy.zpf_lines:
        phases = zpf_line.stable_phases_with_multiplicity
        phase_data = {}
        for p in phases:
            x_data = zpf_line.get_var_list(_get_phase_specific_variable(p, x))
            y_data = zpf_line.get_var_list(_get_phase_specific_variable(p, y))
            phase_data[p] = {
                'x': x_data,
                'y': y_data,
                }
        zpf_data.append(phase_data)
    return zpf_data