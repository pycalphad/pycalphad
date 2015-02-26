"""
This module deals with managing the result of an equilibrium calculation.
"""

from operator import itemgetter

class SublatticeResult(object):
    def __init__(self):
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
        self.mole_fractions = dict()
        self.sublattices = list()

    def __repr__(self):
        return '{0!s}({1!r})'.format(self.__class__, self.__dict__)

    def __str__(self):
        res = '{0:<}#{1}    {2:E}\n'.format(self.name,\
            self.multiplicity, self.volume_fraction)
        for idx, sublattice in enumerate(self.sublattices):
            res += 'Sublattice {0}:\n    {1!s}\n'.format(idx+1, sublattice)
        return res

class EquilibriumResult(object):
    def __init__(self):
        self.components = list()
        self.conditions = list()
        self.mole_fractions = dict()
        self.phases = list()
        self.potentials = dict()

    def __repr__(self):
        return '{0!s}({1!r})'.format(self.__class__, self.__dict__)

    def __str__(self):
        res = ''
        res += 'Enforced Conditions:\n{0!s}\n'.format(self.conditions)
        res += 'Potentials:\n'
        res += '    '.join(['{0!s}={1}'.format(k, v) \
            for k, v in self.potentials.items()])
        res += '\n'
        res += '\n'.join([str(ph) for ph in self.phases])
        return res
