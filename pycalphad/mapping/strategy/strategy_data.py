"""
Data classes to hold outputs from map strategies

- ``SinglePhaseData`` - holds x, y coordinates for a given phase
- ``StrategyData`` - holds a list of ``SinglePhaseData`` with some functions to loop
  over phases, x and y in each ``SinglePhaseData`` object and to get x and y limits
- ``PhaseRegionData`` - alias of ``StrategyData``. This is done to clarify what
  ``BinaryStrategy.get_zpf_data``, ``BinaryStrategy.get_invariant_data``
  ``TernaryStrategy.get_zpf_data``, ``TernaryStrategy.get_invariant_data`` and
  ``IsoplethStrategy.get_invariant_data`` does compared to ``StepStrategy.get_data``
  and ``IsoplethStrategy.get_zpf_data``
"""
import copy
from typing import Union
from dataclasses import dataclass, field

import numpy as np

from pycalphad import variables as v
from pycalphad.mapping.primitives import _get_phase_specific_variable

@dataclass
class SinglePhaseData:
    phase: str
    x: float | list[float]
    y: float | list[float]

@dataclass
class StrategyData:
    data: list[SinglePhaseData]
    xlim: list[float] = field(init=False)
    ylim: list[float] = field(init=False)

    def __post_init__(self):
        all_x = np.concatenate([np.atleast_1d(d.x) for d in self.data], axis=0)
        all_y = np.concatenate([np.atleast_1d(d.y) for d in self.data], axis=0)
        self.xlim = [np.amin(all_x[~np.isnan(all_x)]), np.amax(all_x[~np.isnan(all_x)])]
        self.ylim = [np.amin(all_y[~np.isnan(all_y)]), np.amax(all_y[~np.isnan(all_y)])]

    def __getitem__(self, key: str) -> SinglePhaseData:
        phases = [d.phase for d in self.data]
        if key in phases:
            return self.data[phases.index(key)]
        else:
            raise KeyError(f"{key} not in dataset.")

    @property
    def phases(self):
        return [d.phase for d in self.data]

    @property
    def x(self):
        return [d.x for d in self.data]

    @property
    def y(self):
        return [d.y for d in self.data]

PhaseRegionData = StrategyData


# Helper functions to grab data from tieline strategies (BinaryStrategy and TernaryStrategy)
# This is to avoid repeating get_XXX_data functions for the two strategies
# TODO: Once BinaryStrategy and TernaryStrategy are merged into a more general TielineStrategy,
#       these two functions can be moved into TieLineStrategy
def get_invariant_data_from_tieline_strategy(strategy, x: v.StateVariable, y: v.StateVariable, global_x = False, global_y = False) -> list[PhaseRegionData]:
    """
    Create a dictionary of data for node plotting in binary and ternary plots.

    Parameters
    ----------
    strategy : BinaryStrategy or TernaryStrategy
        The strategy used for generating the data.
    x : v.StateVariable
        The state variable to be used for the x-axis.
    y : v.StateVariable
        The state variable to be used for the y-axis.

    Returns
    -------
    list of dict
        A list where each dictionary contains the following structure::

        {
            "phases": list of str,
            "x": list of float,
            "y": list of float
        }

        The indices in `x` and `y` match the indices in `phases`.
    """
    if hasattr(x, 'phase_name') and x.phase_name is None:
        if not global_x:
            x = copy.deepcopy(x)
            x.phase_name = '*'

    if hasattr(y, 'phase_name') and y.phase_name is None:
        if not global_y:
            y = copy.deepcopy(y)
            y.phase_name = '*'

    invariant_data = []
    for node in strategy.node_queue.nodes:
        # Nodes in binary and ternary mappings are always 3 composition sets
        if len(node.stable_composition_sets) == 3:
            node_phases = node.stable_phases_with_multiplicity
            data = []
            for p in node_phases:
                x_data = node.get_property(_get_phase_specific_variable(p, x))
                y_data = node.get_property(_get_phase_specific_variable(p, y))
                data.append(SinglePhaseData(p, x_data, y_data))
            invariant_data.append(StrategyData(data))

    return invariant_data

def get_tieline_data_from_tieline_strategy(strategy, x: v.StateVariable, y: v.StateVariable, global_x = False, global_y = False) -> list[PhaseRegionData]:
    """
    Create a dictionary of data for plotting ZPF lines.

    Parameters
    ----------
    strategy : BinaryStrategy or TernaryStrategy
        The strategy used for generating the data.
    x : v.StateVariable
        The state variable to be used for the x-axis.
    y : v.StateVariable
        The state variable to be used for the y-axis.

    Returns
    -------
    list of dict
        A list where each dictionary has the following structure::

        {
            "<phase_name>": {
                "x": list of float,
                "y": list of float
            }
        }

        The lengths of the "x" and "y" lists should be equal for each phase in a ZPFLine.
    """
    if hasattr(x, 'phase_name') and x.phase_name is None:
        if not global_x:
            x = copy.deepcopy(x)
            x.phase_name = '*'

    if hasattr(y, 'phase_name') and y.phase_name is None:
        if not global_y:
            y = copy.deepcopy(y)
            y.phase_name = '*'

    zpf_data = []
    for zpf_line in strategy.zpf_lines:
        phases = zpf_line.stable_phases_with_multiplicity
        data = []
        for p in phases:
            x_data = zpf_line.get_var_list(_get_phase_specific_variable(p, x))
            y_data = zpf_line.get_var_list(_get_phase_specific_variable(p, y))
            data.append(SinglePhaseData(p, x_data, y_data))
        zpf_data.append(StrategyData(data))

    return zpf_data