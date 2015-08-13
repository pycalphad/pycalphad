"""
The energy_surf module contains a routine for calculating the
energy surface of a system.
"""

from __future__ import division
from pycalphad import Model
from pycalphad.model import DofError
from pycalphad.core.utils import make_callable, point_sample, generate_dof
from pycalphad.core.utils import endmember_matrix, unpack_kwarg
from pycalphad.log import logger
import pycalphad.variables as v
from sympy import Symbol
import pandas as pd
import numpy as np
import itertools
import collections
import warnings

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
        Model attribute to sample.
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
    warnings.warn('Use pycalphad.calculate() instead', DeprecationWarning, stacklevel=2)
    # Here we check for any keyword arguments that are special, i.e.,
    # there may be keyword arguments that aren't state variables
    pdens_dict = unpack_kwarg(kwargs.pop('pdens', 2000), default_arg=2000)
    model_dict = unpack_kwarg(kwargs.pop('model', Model), default_arg=Model)
    callable_dict = unpack_kwarg(kwargs.pop('callables', None), default_arg=None)

    # Convert keyword strings to proper state variable objects
    # If we don't do this, sympy will get confused during substitution
    statevar_dict = \
        collections.OrderedDict((v.StateVariable(key), value) \
                                for (key, value) in sorted(kwargs.items()))

    # Generate all combinations of state variables for 'map' calculation
    # Wrap single values of state variables in lists
    # Use 'kwargs' because we want state variable names to be stringified
    statevar_values = [_listify(val) for val in statevar_dict.values()]
    statevars_to_map = np.array(list(itertools.product(*statevar_values)))

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

        # Construct an ordered list of the variables
        variables, sublattice_dof = generate_dof(phase_obj, mod.components)

        site_ratios = list(phase_obj.sublattices)
        # Build the "fast" representation of that model
        if callable_dict[phase_name] is None:
            # As a last resort, treat undefined symbols as zero
            # But warn the user when we do this
            # This is consistent with TC's behavior
            undefs = list(out.atoms(Symbol) - out.atoms(v.StateVariable))
            for undef in undefs:
                out = out.xreplace({undef: float(0)})
                logger.warning('Setting undefined symbol %s for phase %s to zero',
                               undef, phase_name)
            comp_sets[phase_name] = make_callable(out, \
                list(statevar_dict.keys()) + variables, mode=mode)
        else:
            comp_sets[phase_name] = callable_dict[phase_name]

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
        # Broadcast compositions and state variables along orthogonal axes
        # This lets us eliminate an expensive Python loop
        data_dict[output] = \
            comp_sets[phase_name](*itertools.chain(
                np.transpose(statevars_to_map[:, :, np.newaxis], (1, 2, 0)),
                np.transpose(points[:, :, np.newaxis], (1, 0, 2)))).T.ravel()
        # Save state variables, with values indexed appropriately
        statevar_vals = np.repeat(statevars_to_map, len(points), axis=0).T
        data_dict.update({str(statevar): vals for statevar, vals \
            in zip(statevar_dict.keys(), statevar_vals)})

        # Map the internal degrees of freedom to global coordinates

        # Normalize site ratios by the sum of site ratios times a factor
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
            avector = [float(vxx.species == comp) * \
                site_ratios[vxx.sublattice_index] for vxx in variables]
            data_dict['X('+comp+')'] = np.tile(np.divide(np.dot(
                points[:, :], avector), site_ratio_normalization),
                                               statevars_to_map.shape[0])

        # Copy coordinate information into data_dict
        # TODO: Is there a more memory-efficient way to deal with this?
        # Perhaps with hierarchical indexing...
        var_fmt = 'Y({0},{1},{2})'
        data_dict.update({var_fmt.format(vxx.phase_name, vxx.sublattice_index,
                                         vxx.species): \
            np.tile(vals, statevars_to_map.shape[0]) \
            for vxx, vals in zip(variables, points.T)})
        all_phase_data.append(pd.DataFrame(data_dict))

    # all_phases_data now contains energy surface information for the system
    return pd.concat(all_phase_data, axis=0, join='outer', \
                            ignore_index=True, verify_integrity=False)
