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
                pdens=1000, **kwargs):
    """
    Sample the energy surface of a system containing the specified
    components and phases. Model parameters are taken from 'dbf' and any
    state variables (T, P, etc.) can be specified as keyword arguments.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list
        Names of phases to consider in the calculation.
    pdens : int, a dict of phase names to int, or a list of both, optional
        Number of points to sample per degree of freedom.

    Returns
    -------
    DataFrame of the energy as a function of composition, temperature, etc.

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

    pdens_dict = collections.defaultdict(lambda: 1000)
    # pdens is a dict of phase names
    if isinstance(pdens, collections.Mapping):
        pdens_dict = pdens
    # pdens is a list containing a dict and an int to be used as a default
    elif isinstance(pdens, collections.Iterable):
        for element in pdens:
            if isinstance(element, collections.Mapping):
                pdens_dict.update(element)
            elif isinstance(element, int):
                # element=element syntax to silence var-from-loop warning
                pdens_dict = collections.defaultdict(
                    lambda element=element: element, pdens_dict)
    else:
        pdens_dict = collections.defaultdict(lambda: pdens)

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
            list(statevar_dict.keys()) + variables)

        # Get the site ratios in each sublattice
        site_ratios = list(phase_obj.sublattices)

        # Sample composition space
        points = point_sample(sublattice_dof, pdof=pdens_dict[phase_name])
        # Generate input d.o.f matrix for all state variable combinations

        # Allocate a contiguous block of memory to store the energies
        energies = np.zeros(len(statevars_to_map) * len(points))
        statevar_list = [None for x in range(len(statevars_to_map) * len(points))]
        # Calculate energies from input matrix
        # We don't construct the entire input matrix at once for memory reasons
        for idx, statevars in enumerate(statevars_to_map):
            start_idx = idx * len(points)
            end_idx = (idx + 1) * len(points)
            inputs = np.array([np.concatenate((list(statevars.values()), point)) \
                                for point in points])
            energies[start_idx:end_idx] = comp_sets[phase_name](*inputs.T)
            statevar_list[start_idx:end_idx] = \
                np.repeat(list(statevars.values()), len(points))

        # Add points and calculated energies to the DataFrame
        data_dict = {'GM':energies, 'Phase':phase_name}
        for statevar in kwargs.keys():
            data_dict[statevar] = statevar_list

        # Map the internal degrees of freedom to global coordinates

        # Normalize site ratios
        # Normalize by the sum of site ratios times a factor
        # related to the site fraction of vacancies
        site_ratio_normalization = np.zeros(len(points))
        for idx, sublattice in enumerate(phase_obj.constituents):
            vacancy_column = np.ones(len(points))
            if 'VA' in set(sublattice):
                var_idx = variables.index(v.SiteFraction(phase_name, idx, 'VA'))
                vacancy_column -= points[:, var_idx]
            site_ratio_normalization += site_ratios[idx] * vacancy_column

        for comp in sorted(comps):
            if comp == 'VA':
                continue
            avector = [float(cur_var.species == comp) * \
                site_ratios[cur_var.sublattice_index] for cur_var in variables]
            data_dict['X('+comp+')'] = np.tile(np.divide(np.dot(
                points[:, :], avector), site_ratio_normalization),
                                               len(statevars_to_map))

        # Copy coordinate information into data_dict
        # TODO: Is there a more memory-efficient way to deal with this?
        # Perhaps with hierarchical indexing...
        #for column_idx, data in enumerate(inputs.T[len(statevar_dict):]):
        #    data_dict[str(variables[column_idx])] = \
        #        pd.Series(data, dtype='float16')

        all_phase_data.append(pd.DataFrame(data_dict))

    # all_phases_data now contains energy surface information for the system
    return pd.concat(all_phase_data, axis=0, join='outer', \
                            ignore_index=True, verify_integrity=False), \
                            comp_sets
