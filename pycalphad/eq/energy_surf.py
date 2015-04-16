"""
The energy_surf module contains a routine for calculating the
energy surface of a system.
"""

from __future__ import division
from pycalphad import Model
from pycalphad.model import DofError
from pycalphad.eq.utils import make_callable, point_sample, generate_dof
from pycalphad.eq.utils import endmember_matrix, unpack_kwarg
from pycalphad.log import logger
import pycalphad.variables as v
from sympy import Symbol
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
    # for debugging purposes; return input (do nothing)
    if max_iterations < 0:
        return input_matrix, energies
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
    # Filter for real simplices
    simplices = hull.simplices[hull.equations[:, -1] <= -1e-6]
    vertices = list(set(np.asarray(simplices).ravel()))
    del global_matrix
    # terminating condition
    if max_iterations == 0:
        return input_matrix[vertices, :], energies[vertices]
    # For the simplices on the hull, calculate the centroids in internal dof
    centroid_matrix = input_matrix[np.asarray(simplices).ravel()]
    centroid_matrix.shape = (len(simplices), len(simplices[0]),
                             len(input_matrix[0]))
    centroid_matrix = np.mean(centroid_matrix, axis=1, dtype=np.float64)

    # Calculate energies of the centroid points
    centroid_energies = energy_func(*centroid_matrix.T)
    # Group together the old points and new points
    input_matrix = np.concatenate((input_matrix[vertices, :],
                                   centroid_matrix), axis=0)
    energies = np.concatenate((energies[vertices], centroid_energies),
                              axis=0)
    # Save some memory since we already grouped these
    del centroid_matrix
    del centroid_energies

    # Call recursively for next iteration, decrementing max_iterations
    return refine_energy_surf(input_matrix, energies,
                              phase_obj, comps, variables,
                              energy_func, max_iterations=max_iterations-1)

def energy_surf(dbf, comps, phases, mode=None, output='GM', **kwargs):
    """
    Sample the property surface of 'output' containing the specified
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
    mode : string, optional
        See 'make_callable' docstring for details.
    output : string, optional
        Model attribute to sample. Default is 'energy'.
    pdens : int, a dict of phase names to int, or a list of both, optional
        Number of points to sample per degree of freedom.
    model : Model, a dict of phase names to Model, or a list of both, optional
        Model class to use for each phase.

    Returns
    -------
    DataFrame of the output as a function of composition, temperature, etc.

    Examples
    --------
    None yet.
    """
    # Here we check for any keyword arguments that are special, i.e.,
    # there may be keyword arguments that aren't state variables
    pdens_dict = unpack_kwarg(kwargs.pop('pdens', 2000), default_arg=2000)
    model_dict = unpack_kwarg(kwargs.pop('model', Model), default_arg=Model)

    # Convert keyword strings to proper state variable objects
    # If we don't do this, sympy will get confused during substitution
    statevar_dict = \
        dict((v.StateVariable(key), value) \
             for (key, value) in kwargs.items())

    # Generate all combinations of state variables for 'map' calculation
    # Wrap single values of state variables in lists
    # Use 'kwargs' because we want state variable names to be stringified
    statevar_values = [_listify(val) for val in kwargs.values()]
    statevars_to_map = [dict(zip(kwargs.keys(), prod)) \
        for prod in itertools.product(*statevar_values)]

    # Consider only the active phases
    active_phases = dict((name.upper(), dbf.phases[name.upper()]) \
        for name in phases)
    comp_sets = {}
    # Construct a list to hold all the data
    all_phase_data = []
    for phase_name, phase_obj in sorted(active_phases.items()):
        # Build the symbolic representation of the energy
        mod = model_dict[phase_name]
        # if this is an object type, we need to construct it
        if isinstance(mod, type):
            try:
                mod = mod(dbf, comps, phase_name)
            except DofError:
                # we can't build the specified phase because the
                # specified components aren't found in every sublattice
                # we'll just skip it
                logger.warning("""Suspending specified phase %s due to
                some sublattices containing only unspecified components""",
                               phase_name)
                continue
        try:
            out = getattr(mod, output)
        except AttributeError:
            raise AttributeError('Missing Model attribute {0} specified for {1}'
                                 .format(output, mod.__class__))
        # As a last resort, treat undefined symbols as zero
        # But warn the user when we do this
        # This is consistent with TC's behavior
        undefs = list(out.atoms(Symbol) - out.atoms(v.StateVariable))
        for undef in undefs:
            out = out.xreplace({undef: float(0)})
            logger.warning('Setting undefined symbol %s for phase %s to zero',
                           undef, phase_name)
        # Construct an ordered list of the variables
        variables, sublattice_dof = generate_dof(phase_obj, mod.components)

        # Build the "fast" representation of that model
        comp_sets[phase_name] = make_callable(out, \
            list(statevar_dict.keys()) + variables, mode=mode)

        # Get the site ratios in each sublattice
        site_ratios = list(phase_obj.sublattices)

        # Eliminate pure vacancy endmembers from the calculation
        vacancy_indices = list()
        for idx, sublattice in enumerate(phase_obj.constituents):
            if 'VA' in sorted(sublattice) and 'VA' in sorted(comps):
                vacancy_indices.append(sorted(sublattice).index('VA'))
        if len(vacancy_indices) != len(phase_obj.constituents):
            vacancy_indices = None
        logger.debug('vacancy_indices: %s', vacancy_indices)
        # Add all endmembers to guarantee their presence
        points = endmember_matrix(sublattice_dof,
                                  vacancy_indices=vacancy_indices)

        # Sample composition space for more points
        if sum(sublattice_dof) > len(sublattice_dof):
            points = np.concatenate((points,
                                     point_sample(sublattice_dof,
                                                  pdof=pdens_dict[phase_name])
                                    ))



        # If there are nontrivial sublattices with vacancies in them,
        # generate a set of points where their fraction is zero and renormalize
        for idx, sublattice in enumerate(phase_obj.constituents):
            if 'VA' in set(sublattice) and len(sublattice) > 1:
                var_idx = variables.index(v.SiteFraction(phase_name, idx, 'VA'))
                addtl_pts = np.copy(points)
                # set vacancy fraction to log-spaced between 1e-10 and 1e-6
                addtl_pts[:, var_idx] = np.power(10.0, -10.0*(1.0 - addtl_pts[:, var_idx]))
                # renormalize site fractions
                cur_idx = 0
                for ctx in sublattice_dof:
                    end_idx = cur_idx + ctx
                    addtl_pts[:, cur_idx:end_idx] /= \
                        addtl_pts[:, cur_idx:end_idx].sum(axis=1)[:, None]
                    cur_idx = end_idx
                # add to points matrix
                points = np.concatenate((points, addtl_pts), axis=0)

        data_dict = {'Phase': phase_name}
        # Generate input d.o.f matrix for all state variable combinations
        for statevars in statevars_to_map:
            # Prefill the state variable arguments to the energy function
            energy_func = \
                lambda *args: comp_sets[phase_name](
                    *itertools.chain(list(statevars.values()),
                                     args))
            # Get the stable points and energies for this configuration
            # Set max refinements equal to the number of independent dof
            mxr = sum(phase_obj.sublattices) - len(phase_obj.sublattices)
            refined_points, energies = \
                refine_energy_surf(points, None, phase_obj, comps,
                                   variables, energy_func, max_iterations=-1)
            try:
                data_dict[output].extend(energies)
                for statevar in kwargs.keys():
                    data_dict[statevar].extend(
                        list(np.repeat(list(statevars.values()),
                                       len(refined_points))))
            except KeyError:
                data_dict[output] = list(energies)
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
            try:
                for column_idx, data in enumerate(refined_points.T):
                    data_dict[str(variables[column_idx])].extend(list(data))
            except KeyError:
                for column_idx, data in enumerate(refined_points.T):
                    data_dict[str(variables[column_idx])] = list(data)

        all_phase_data.append(pd.DataFrame(data_dict))

    # all_phases_data now contains energy surface information for the system
    return pd.concat(all_phase_data, axis=0, join='outer', \
                            ignore_index=True, verify_integrity=False)
