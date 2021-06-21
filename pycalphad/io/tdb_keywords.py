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
    'AMEND_PHASE_DESCRIPTION',
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
    # Gibbs energy parameters
    'G',      # Gibbs energy
    'L',      # Excess Gibbs energy
    # Physical model parameters
    'TC',     # Curie temperature
    'NT',     # Neel temperature
    'BMAGN',  # Bohr magneton number
    'GD',     # Gibbs energy difference between liquid and amorphous states
    'THETA',  # Einstein temperature (log)
    # Molar volume parameters
    'V0',     # Molar volume at STP
    'VA',     # Integrated thermal expansivity
    'VC',     # High-pressure fitting parameter
    'VK',     # Isothermal compressibility
    # Property model parameters
    'VISC',   # Viscosity, RT*log(viscosity)
    'ELRS',   # Electric resistivity
    'THCD',   # Thermal Conductivity
    'SIGM',   # Surface tension of a liquid endmember
    'XI',     # Surface tension dampening factor for a constituent
    # Mobility parameters
    'MQ',     # Activation enthalpy for mobility
    'MF',     # Pre-exponential factor for mobility
    'DQ',     # Activation enthalpy for diffusivity
    'DF',     # Pre-expontential factor for diffusivity
    'VS',     # Volume per mole of volume-carrying species
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
