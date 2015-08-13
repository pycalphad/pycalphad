"""
This module deals with managing the result of an equilibrium calculation.
"""

from operator import itemgetter
from collections import defaultdict
import copy
import pycalphad.variables as v

class SublatticeResult(object):
    def __init__(self):
        self.site_count = None
        self.site_fractions = dict()

    def __repr__(self):
        return '{0!s}({1!r})'.format(self.__class__, self.__dict__)

    def __str__(self):
        return '    '.join(['{0:<} {1:E}'.format(k, v) \
            for k, v in sorted(self.site_fractions.items(), key=itemgetter(0))])

class PhaseResult(object):
    def __init__(self):
        self.name = None
        self.volume_fraction = None
        self.multiplicity = None
        self.sublattices = list()

    @property
    def mole_fractions(self):
        result = defaultdict(lambda: 0.0)
        site_ratio_normalization = 0.0
        # Calculate normalization factor
        for sublattice in self.sublattices:
            if 'VA' in set(sublattice.site_fractions.keys()):
                site_ratio_normalization += sublattice.site_count * \
                    (1.0 - sublattice.site_fractions['VA'])
            else:
                site_ratio_normalization += sublattice.site_count
        site_ratios = [c.site_count/site_ratio_normalization \
            for c in self.sublattices]
        # Sum up site fraction contributions
        for idx, sublattice in enumerate(self.sublattices):
            for component in sublattice.site_fractions.keys():
                if component == 'VA':
                    continue
                result[component] += site_ratios[idx] * \
                    sublattice.site_fractions[component]
        return result

    def __repr__(self):
        return '{0!s}({1!r})'.format(self.__class__, self.__dict__)

    def __str__(self):
        res = '{0:<}#{1}    {2:E}\n'.format(self.name,\
            self.multiplicity, self.volume_fraction)
        res += 'Mole fractions:    '
        res += '    '.join(['{0:<} {1:E}'.format(k, v) \
            for k, v in sorted(self.mole_fractions.items(), key=itemgetter(0))])
        res += '\n'
        for idx, sublattice in enumerate(self.sublattices):
            res += 'Sublattice {0}: {1} sites\n    {2!s}\n'.format(idx+1, \
                sublattice.site_count, sublattice)
        return res

class EquilibriumResult(object):
    def __init__(self, phases, components, potentials, energy, variables):
        self.phases = list()
        self.energy = energy
        self.components = components
        self.potentials = potentials
        phase_res = PhaseResult()
        for variable, value in variables:
            if isinstance(variable, v.PhaseFraction):
                # New phase: Append old one if not empty
                if phase_res.name is not None:
                    self.phases.append(copy.deepcopy(phase_res))
                phase_res = PhaseResult()
                phase_res.name = variable.phase_name
                phase_res.multiplicity = variable.multiplicity
                phase_res.volume_fraction = value
            elif isinstance(variable, v.SiteFraction):
                # Add sublattices if this variable has a larger index
                if variable.sublattice_index >= len(phase_res.sublattices):
                    phase_res.sublattices.extend(SublatticeResult() \
                        for i in range(variable.sublattice_index - \
                            len(phase_res.sublattices) + 1))
                phase_res.sublattices[variable.sublattice_index].site_count = \
                    phases[variable.phase_name].sublattices[
                        variable.sublattice_index]
                phase_res.sublattices[
                    variable.sublattice_index
                    ].site_fractions[variable.species] = value
        # Add final phase
        self.phases.append(copy.deepcopy(phase_res))

    @property
    def mole_fractions(self):
        result = defaultdict(lambda: 0.0)
        # Sum up phase contributions
        for phase in self.phases:
            for component, value in phase.mole_fractions.items():
                result[component] += phase.volume_fraction * value
        return result

    def __repr__(self):
        return '{0!s}({1!r})'.format(self.__class__, self.__dict__)

    def __str__(self):
        res = ''
        res += 'Molar Gibbs Energy:    {0:E}\n'.format(self.energy)
        res += 'Potentials:\n'
        res += '    '.join(['{0!s}={1}'.format(k, v) \
            for k, v in self.potentials.items()])
        res += '\n'
        res += 'Molar Composition:\n'
        res += '    '.join(['X({0!s})={1:E}'.format(k, v) \
            for k, v in sorted(list(self.mole_fractions.items()))])
        res += '\n'
        res += '\n'
        res += '\n'.join([str(ph) for ph in self.phases])
        return res
