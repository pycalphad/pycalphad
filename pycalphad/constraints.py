"""
The constraints module contains definitions for equilibrium constraints and
their Jacobian.
"""

import pycalphad.variables as v
import numpy as np

# An index range is a list of (ordered pairs of indices).
def sitefrac_cons(input_x, idx_range):
    """
    Accepts input vector and index range and returns
    site fraction constraint.
    """
    return sum(input_x[idx_range[0]:idx_range[1]]) - 1

def sitefrac_jac(input_x, idx_range):
    """
    Accepts input vector and index range and returns
    Jacobian of site fraction constraint.
    """
    output_x = np.zeros(len(input_x))
    for idx in range(idx_range[0], idx_range[1]):
        output_x[idx] = 1
    return output_x

def molefrac_cons(input_x, species, fix_val, all_variables, phases):
    """
    Accept input vector, species and fixed value.
    Returns constraint.
    """
    output = -fix_val
    phase_idx = 0
    site_ratios = []
    for idx, variable in enumerate(all_variables):
        if isinstance(variable, v.PhaseFraction):
            phase_idx = idx
            # Normalize site ratios
            site_ratio_normalization = 0
            for n_idx, sublattice in \
                enumerate(phases[variable.phase_name].constituents):
                if species in set(sublattice):
                    site_ratio_normalization += \
                        phases[variable.phase_name].sublattices[n_idx]

            site_ratios = [c/site_ratio_normalization for c in \
                phases[variable.phase_name].sublattices]
        if isinstance(variable, v.SiteFraction) and \
            species == variable.species:
            output += input_x[phase_idx] * \
                site_ratios[variable.sublattice_index] * input_x[idx]
    return output

#pylint: disable-msg=W0613
def molefrac_jac(input_x, species, fix_val, all_variables, phases):
    """
    Accept input vector, species and fixed value.
    Returns Jacobian of constraint.
    """
    output_x = np.zeros(len(input_x))
    phase_idx = 0
    site_ratios = []
    for idx, variable in enumerate(all_variables):
        if isinstance(variable, v.PhaseFraction):
            phase_idx = idx
            # Normalize site ratios
            site_ratio_normalization = 0
            for n_idx, sublattice in \
                enumerate(phases[variable.phase_name].constituents):
                if species in set(sublattice):
                    site_ratio_normalization += \
                        phases[variable.phase_name].sublattices[n_idx]

            site_ratios = [c/site_ratio_normalization \
                for c in phases[variable.phase_name].sublattices]
            # We add the phase fraction Jacobian contribution below
        if isinstance(variable, v.SiteFraction) and \
            species == variable.species:
            output_x[idx] += input_x[phase_idx] * \
                site_ratios[variable.sublattice_index]
            output_x[phase_idx] += input_x[idx] * \
                site_ratios[variable.sublattice_index]
    return output_x
