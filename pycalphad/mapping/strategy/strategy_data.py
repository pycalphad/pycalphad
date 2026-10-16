"""
Data classes to hold outputs from map strategies

- ``SinglePhaseData`` - holds x, y coordinates for a given phase
- ``StrategyData`` - holds a list of ``SinglePhaseData`` with some functions to loop
  over phases, x and y in each ``SinglePhaseData`` object and to get x and y limits
- ``PhaseRegionData`` - alias of ``StrategyData``. This is done to clarify what
  ``TielineStrategy.get_tieline_data``, ``TielineStrategy.get_invariant_data`` and
  ``IsoplethStrategy.get_invariant_data`` does compared to ``StepStrategy.get_data``
  and ``IsoplethStrategy.get_zpf_data``
"""
from typing import Union
from dataclasses import dataclass, field

import numpy as np


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
