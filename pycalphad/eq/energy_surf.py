"""
The energy_surf module contains a routine for calculating the
energy surface of a system.
"""

from __future__ import division
from pycalphad import Model
from pycalphad.eq.utils import make_callable, point_sample, generate_dof
import pycalphad.variables as v
import pandas as pd
import numpy as np
import itertools
import collections

try:
    set
except NameError:
    from sets import Set as set #pylint: disable=W0622

def _listify(val):
    "Return val if val is iterable; otherwise return list(val)."
    if isinstance(val, collections.Iterable):
        return val
    else:
        return [val]

#pylint: disable=W0142
def energy_surf(dbf, comps, phases,
                points_per_phase=100, ast='numpy', **kwargs):
    """
    Calculate the energy surface of a system containing the specified
    components and phases. Model parameters are taken from 'db' and any
    state variables (T, P, etc.) can be specified as keyword arguments.
    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names (case-sensitive) of components to consider in the calculation.
    phases : list
        Names (case-sensitive) of phases to consider in the calculation.
    points_per_phase : int, optional
        Approximate number of points to sample per phase.
    ast : ['numpy'], optional
        Specify how we should construct the callable for the energy.

    Returns
    -------
    DataFrame of the energy surface.

    Examples
    --------
    None yet.
    """
    # Here we would check for any keyword arguments that are special, i.e.,
    # there may be keyword arguments that aren't state variables

    # Convert keyword strings to proper state variable objects
    # If we don't do this, sympy will get confused during substitution
    statevar_dict = \
        dict((v.StateVariable(key), value) for (key, value) in kwargs.items())
    # Generate all combinations of state variables for 'map' calculation
    # Wrap single values of state variables in lists
    # Use 'kwargs' because we want state variable names to be stringified
    statevar_values = [_listify(val) for val in kwargs.values()]
    statevars_to_map = [dict(zip(kwargs.keys(), prod)) \
        for prod in itertools.product(*statevar_values)]

    active_comps = set(comps)
    # Consider only the active phases
    active_phases = dict((name.upper(), dbf.phases[name.upper()]) \
        for name in phases)
    comp_sets = {}
    # Construct a list to hold all the data
    all_phase_data = []
    for phase_name, phase_obj in active_phases.items():
        # Build the symbolic representation of the energy
        mod = Model(dbf, comps, phase_name)
        # Construct an ordered list of the variables
        variables, sublattice_dof = generate_dof(phase_obj, active_comps)

        # Build the "fast" representation of that model
        comp_sets[phase_name] = make_callable(mod.ast, \
            list(statevar_dict.keys()) + variables, mode=ast)

        # Calculate the number of components in each sublattice
        nontrivial_sublattices = len(sublattice_dof) - sublattice_dof.count(1)
        # Get the site ratios in each sublattice
        site_ratios = list(phase_obj.sublattices)

        # Normalize site ratios
        site_ratio_normalization = 0
        for idx, sublattice in enumerate(phase_obj.constituents):
            # sublattices with only vacancies don't count
            if len(sublattice) == 1 and sublattice[0] == 'VA':
                continue
            site_ratio_normalization += site_ratios[idx]

        site_ratios = [c/site_ratio_normalization for c in site_ratios]
        # Choose a sensible number of compositions to sample
        num_points = None
        if nontrivial_sublattices > 0:
            num_points = int(points_per_phase**(1/nontrivial_sublattices))
        else:
            # Fixed stoichiometry
            num_points = 1

        # Sample composition space
        points = point_sample(sublattice_dof, size=points_per_phase)
        # Generate input d.o.f matrix for all state variable combinations
        statevar_values = [list(statevars.values()) \
            for statevars in statevars_to_map]
        inputs_prod = list(itertools.product(statevar_values, points))
        inputs = np.asarray([np.concatenate(x) for x in inputs_prod])

        # Calculate energies from input matrix
        energies = comp_sets[phase_name](*inputs.T)

        # Add points and calculated energies to the DataFrame
        data_dict = {'GM':energies, 'Phase':phase_name}
        data_dict.update(dict(zip(kwargs.keys(),
                                  inputs.T[0:len(statevar_dict)])))

        # Map the internal degrees of freedom to global coordinates
        for comp in sorted(comps):
            if comp == 'VA':
                continue
            avector = [float(cur_var.species == comp) * \
                site_ratios[cur_var.sublattice_index] \
                for cur_var in variables]
            data_dict['X('+comp+')'] = np.dot(inputs[:, len(statevar_dict):], \
            avector)

        # Copy coordinate information into data_dict
        for column_idx, data in enumerate(inputs.T[len(statevar_dict):]):
            data_dict[str(variables[column_idx])] = data

        all_phase_data.append(pd.DataFrame(data_dict))

    # all_phases_data now contains energy surface information for the system
    return pd.concat(all_phase_data, axis=0, join='outer', \
                            ignore_index=True, verify_integrity=False), comp_sets
    #return pd.DataFrame(all_phase_data), comp_sets
