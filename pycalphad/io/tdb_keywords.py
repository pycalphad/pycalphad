"""
This module defines valid Thermo-Calc TDB keywords and handles abbreviations.
Note that not all of these keywords are fully supported yet.
"""

import re

# Reference: Thermo-Calc Database Manager's Guide
TDB_KEYWORDS = sorted([
    'ELEMENT',
    'SPECIES',
    'PHASE',
    'CONSTITUENT',
    'ADD_CONSTITUENT',
    'COMPOUND_PHASE',
    'ALLOTROPIC_PHASE',
    'TEMPERATURE_LIMITS',
    'DEFINE_SYSTEM_DEFAULT',
    'DEFAULT_COMMAND',
    'DATABASE_INFORMATION',
    'TYPE_DEFINITION',
    'FTP_FILE',
    'FUNCTION',
    'PARAMETER',
    'OPTIONS',
    'TABLE',
    'ASSESSED_SYSTEMS',
    'REFERENCE_FILE',
    'LIST_OF_REFERENCES',
    'ADD_REFERENCE',
    'CASE',
    'ENDCASE',
    'VERSION_DATA',
    'VERSION_DATE',
    'DIFFUSION',
    'ZERO_VOLUME_SPECIES'
])

# Reference: Thermo-Calc Console Mode Command Reference, Version 4.1
TDB_COMMANDS = sorted([
    'AMEND_PHASE_DESCRIPTION'
    'DEFINE_ELEMENTS'
])

# Reference: Thermo-Calc Console Mode Command Reference, Version 4.1
TDB_PHASE_DESCRIPTIONS = sorted([
    'EXCESS_MODEL',
    'MAGNETIC_ORDERING',
    'DEBYE_HUCKEL',
    'STATUS_BITS',
    'NEW_CONSTITUENT',
    'RENAME_PHASE',
    'COMPOSITION_SETS',
    'GLASS_TRANSITION',
    'DISORDERED_PART',
    'MAJOR_CONSTITUENT',
    'ZRO2_TRANSITION',
    'REMOVE_ADDITION',
    'QUASICHEM_IONIC',
    'QUASICHEM_FACT00',
    'QUASICHEM_IRSID',
    'TERNARY_EXTRAPOLAT',
    'HKF_ELECTROSTATIC',
    'DEFAULT_STABLE',
    'SITE_RATIOS',
    'FRACTION_LIMITS'
])

TDB_PARAM_TYPES = sorted([
    'G',
    'L',
    'TC',
    'BMAGN'
    'MQ',
    'MF',
    'DQ',
    'DF',
    'V0',
    'VS'
])

def expand_keyword(possible, candidate):
    """
    Expand an abbreviated keyword based on the provided list.

    Parameters
    ----------
    possible : list of str
        Possible keywords for 'candidate' to be matched against.
    candidate : str
        Abbreviated keyword to expand.

    Returns
    -------
    list of str of matching expanded keywords

    Examples
    --------
    None yet.
    """
    # Rewritten to escape each token in the split string, instead of the input
    # The reason is that Python 2.7 will escape underscore characters
    candidate_pieces = [re.escape(x) for x in \
        candidate.upper().replace('-', '_').split('_')]
    pattern = r'^'
    pattern += r'[^_\s]*_'.join(candidate_pieces)
    pattern += r'[^\s]*$'
    matches = [re.match(pattern, pxd) for pxd in possible]
    matches = [m.string for m in matches if m is not None]
    if len(matches) == 0:
        raise ValueError('{0} does not match {1}'.format(candidate, possible))

    return matches

if __name__ == '__main__':
    TEST_LIST = [
        'PARAMETER',
        'ELEMENT',
        'CALCULATE_EQUILIBRIUM',
        'CALCULATE_ALL_EQUILIBRIA',
        'LIST_EQUILIBRIUM',
        'LIST_INITIAL_EQUILBRIUM',
        'LOAD_INITIAL_EQUILBRIUM',
        'LIST_PHASE_DATA',
        'SET_ALL_START_VALUES',
        'SET_AXIS_VARIABLE',
        'SET_START_CONSTITUENT',
        'SET_START_VALUE',
        'SET_AXIS_PLOT_STATUS',
        'SET_AXIS_TEXT_STATUS',
        'SET_AXIS_TYPE',
        'SET_OPTIMIZING_CONDITION',
        'SET_OPTIMIZING_VARIABLE',
        'SET_OUTPUT_LEVEL'
    ]
    TEST_INPUT = [
        'Par',
        'Elem',
        'PAR',
        'C-E',
        'C-A',
        'LI-I-E',
        'LO-I-E',
        'L-P-D',
        'S-A-S',
        'S-AL',
        'S-A-V',
        'S-S-C',
        'S-S-V',
        'S-A-P',
        'S-A-T-S',
        'S-A-TE',
        'S-A-TY',
        'S-O-C',
        'S-O-V',
        'S-O-L',
        'S-OU'
    ]
    for inp in TEST_INPUT:
        print('{1} matches {0}'.format(expand_keyword(TEST_LIST, inp), inp))
