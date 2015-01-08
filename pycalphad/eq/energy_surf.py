"""
The energy_surf module contains a routine for calculating the
energy surface of a system.
"""

from __future__ import division
from pycalphad import Model
from pycalphad.eq.utils import make_callable, point_sample, generate_dof
import pycalphad.variables as v
import scipy.spatial
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

def refine_energy_surf(input_matrix, energies, phase_obj, comps, variables,
                       energy_func, max_iterations=1):
    """
    Recursively refine the equilibrium energy surface of a phase, starting
    from some initial points.

    Parameters
    ----------
    input_matrix : ndarray
        Matrix whose rows can be fed into 'energy_callable'
        Note that 'input_matrix' should only contain one set of values for the
        state variables, e.g., fixed T, P.
    energies : ndarray or None
        One-dimensional array containing energies of 'input_matrix'
    phase_obj : Phase
        Phase whose energy surface is being refined
    comps : list of string
        Names of active components
    variables : list of StateVariable
        Ordered list of degrees of freedom in 'input_matrix'
    energy_func : callable
        Function that accepts rows of 'input_matrix' and returns the energy
    max_iterations : int, optional
        Number of recursive refinement iterations.

    Returns
    -------
    tuple of refined_input_matrix, energies
    """
    # If energies is None, calculate energies of input_matrix
    if energies is None:
        energies = energy_func(*input_matrix.T)
    # Normalize site ratios
    # Normalize by the sum of site ratios times a factor
    # related to the site fraction of vacancies
    site_ratio_normalization = np.zeros(len(input_matrix))
    for idx, sublattice in enumerate(phase_obj.constituents):
        vacancy_column = np.ones(len(input_matrix))
        if 'VA' in set(sublattice):
            var_idx = variables.index(v.SiteFraction(phase_obj.name, idx, 'VA'))
            vacancy_column -= input_matrix[:, var_idx]
        site_ratio_normalization += phase_obj.sublattices[idx] * vacancy_column

    comp_list = sorted(list(comps))
    # We aren't going to calculate mole fraction of vacancies
    try:
        comp_list.remove('VA')
    except ValueError:
        pass
    # Remove last component from the list, as it's dependent
    comp_list.pop()
    # Map input_matrix to global coordinates (mole fractions)
    global_matrix = np.zeros((len(input_matrix), len(comp_list)+1))
    for comp_idx, comp in enumerate(comp_list):
        avector = [float(cur_var.species == comp) * \
            phase_obj.sublattices[cur_var.sublattice_index] \
            for cur_var in variables]
        global_matrix[:, comp_idx] = np.divide(np.dot(
            input_matrix[:, :], avector), site_ratio_normalization)
    global_matrix[:, -1] = energies

    # If this is a stoichiometric phase, we can't calculate a hull
    # Just return all points and energies
    if len(global_matrix) < len(comp_list)+1:
        return input_matrix, energies
    # Calculate the convex hull of the energy surface in global coordinates
    hull = scipy.spatial.ConvexHull(global_matrix, qhull_options='QJ')
    del global_matrix
    # terminating condition
    if max_iterations <= 0:
        return input_matrix[hull.vertices, :], energies[hull.vertices]
    # For the simplices on the hull, calculate the centroids in internal dof
    centroid_matrix = input_matrix[np.asarray(hull.simplices).ravel()]
    centroid_matrix.shape = (len(hull.simplices), len(hull.simplices[0]),
                     len(input_matrix[0]))
    centroid_matrix = np.mean(centroid_matrix, axis=1, dtype=np.float64)
    # Calculate energies of the centroid points
    centroid_energies = energy_func(*centroid_matrix.T)
    input_matrix = np.concatenate((input_matrix, centroid_matrix), axis=0)
    energies = np.concatenate((energies, centroid_energies), axis=0)
    del centroid_matrix
    del centroid_energies
    return refine_energy_surf(input_matrix, energies,
                              phase_obj, comps, variables,
                              energy_func, max_iterations=max_iterations-1)

#pylint: disable=W0142
def energy_surf(dbf, comps, phases,
                pdens=1000, mode=None, **kwargs):
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
            list(statevar_dict.keys()) + variables, mode=mode)

        # Get the site ratios in each sublattice
        site_ratios = list(phase_obj.sublattices)

        # Sample composition space
        points = point_sample(sublattice_dof, pdof=pdens_dict[phase_name])

        data_dict = {'Phase': phase_name}
        # Generate input d.o.f matrix for all state variable combinations
        for statevars in statevars_to_map:
            # Prefill the state variable arguments to the energy function
            energy_func = \
                lambda *args: comp_sets[phase_name](
                    *itertools.chain(list(statevars.values()),
                                     args))
            # Get the stable points and energies for this configuration
            refined_points, energies = \
                refine_energy_surf(points, None, phase_obj, comps,
                                   variables, energy_func, max_iterations=2)
            try:
                data_dict['GM'].extend(energies)
                for statevar in kwargs.keys():
                    data_dict[statevar].extend(
                        list(np.repeat(list(statevars.values()),
                                       len(refined_points))))
            except KeyError:
                data_dict['GM'] = list(energies)
                for statevar in kwargs.keys():
                    data_dict[statevar] = \
                        list(np.repeat(list(statevars.values()),
                                       len(refined_points)))

            # Map the internal degrees of freedom to global coordinates

            # Normalize site ratios
            # Normalize by the sum of site ratios times a factor
            # related to the site fraction of vacancies
            site_ratio_normalization = np.zeros(len(refined_points))
            for idx, sublattice in enumerate(phase_obj.constituents):
                vacancy_column = np.ones(len(refined_points))
                if 'VA' in set(sublattice):
                    var_idx = variables.index(v.SiteFraction(phase_name, idx, 'VA'))
                    vacancy_column -= refined_points[:, var_idx]
                site_ratio_normalization += site_ratios[idx] * vacancy_column

            for comp in sorted(comps):
                if comp == 'VA':
                    continue
                avector = [float(cur_var.species == comp) * \
                    site_ratios[cur_var.sublattice_index] for cur_var in variables]
                try:
                    data_dict['X('+comp+')'].extend(list(np.divide(np.dot(
                        refined_points[:, :], avector), site_ratio_normalization)))
                except KeyError:
                    data_dict['X('+comp+')'] = list(np.divide(np.dot(
                        refined_points[:, :], avector), site_ratio_normalization))

        # Copy coordinate information into data_dict
        # TODO: Is there a more memory-efficient way to deal with this?
        # Perhaps with hierarchical indexing...
        #for column_idx, data in enumerate(inputs.T[len(statevar_dict):]):
        #    data_dict[str(variables[column_idx])] = data

        all_phase_data.append(pd.DataFrame(data_dict))

    # all_phases_data now contains energy surface information for the system
    return pd.concat(all_phase_data, axis=0, join='outer', \
                            ignore_index=True, verify_integrity=False)
