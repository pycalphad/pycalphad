"""
The calculate module contains a routine for calculating the
property surface of a system.
"""

from __future__ import division
from pycalphad import Model
from pycalphad.model import DofError
from pycalphad.core.utils import make_callable, point_sample, generate_dof
from pycalphad.core.utils import endmember_matrix, unpack_kwarg
from pycalphad.core.utils import unpack_condition, unpack_phases
from pycalphad.log import logger
import pycalphad.variables as v
from sympy import Symbol
import xray
from xray.core.npcompat import broadcast_to
import numpy as np
import itertools
import collections

try:
    set
except NameError:
    from sets import Set as set #pylint: disable=W0622


def _generate_fake_points(components, statevar_dict, energy_limit, output, maximum_internal_dof):
    """
    Generate points for a fictitious hyperplane used as a starting point for energy minimization.
    """
    coordinate_dict = {'component': components}
    coordinate_dict.update({str(key): value for key, value in statevar_dict.items()})
    largest_energy = float(energy_limit)
    if largest_energy < 0:
        largest_energy *= 0.99
    else:
        largest_energy *= 1.01
    output_columns = [str(x) for x in statevar_dict.keys()] + ['points']
    statevar_shape = tuple(len(np.atleast_1d(x)) for x in statevar_dict.values())
    # The internal dof for the fake points are all NaNs
    expanded_points = np.full(statevar_shape + (len(components), maximum_internal_dof), np.nan)
    data_arrays = {'X': (output_columns + ['component'],
                         broadcast_to(np.eye(len(components)), statevar_shape + (len(components), len(components)))),
                   'Y': (output_columns + ['internal_dof'], expanded_points),
                   'Phase': (output_columns, np.full(statevar_shape + (len(components),), '_FAKE_', dtype='S6')),
                   output: (output_columns, np.full(statevar_shape + (len(components),), largest_energy))
                   }
    return xray.Dataset(data_arrays, coords=coordinate_dict)


def _compute_phase_values(phase_obj, components, variables, statevar_dict,
                          points, func, output, maximum_internal_dof):
    """
    Calculate output values for a particular phase.

    Parameters
    ----------
    phase_obj : Phase
        Phase object from a thermodynamic database.
    components : list
        Names of components to consider in the calculation.
    variables : list
        Names of variables in the phase's internal degrees of freedom.
    statevar_dict : OrderedDict {str -> float or sequence}
        Mapping of state variables to desired values. This will broadcast if necessary.
    points : ndarray
        Inputs to 'func', except state variables. Columns should be in 'variables' order.
    func : callable
        Function of state variables and 'variables'.
        See 'make_callable' docstring for details.
    output : string
        Desired name of the output result in the Dataset.
    maximum_internal_dof : int
        Largest number of internal degrees of freedom of any phase. This is used
        to guarantee different phase's Datasets can be concatenated.

    Returns
    -------
    xray.Dataset of the output attribute as a function of state variables

    Examples
    --------
    None yet.
    """
    # Broadcast compositions and state variables along orthogonal axes
    # This lets us eliminate an expensive Python loop
    statevar_grid = np.meshgrid(*itertools.chain(statevar_dict.values(),
                                                 [np.empty(points.shape[-2])]),
                                sparse=True, indexing='ij')[:-1]
    points = broadcast_to(points, tuple(len(np.atleast_1d(x)) for x in statevar_dict.values()) + points.shape[-2:])
    phase_output = func(*itertools.chain(statevar_grid, np.rollaxis(points, -1, start=0)))

    # Map the internal degrees of freedom to global coordinates
    # Normalize site ratios by the sum of site ratios times a factor
    # related to the site fraction of vacancies
    site_ratio_normalization = np.zeros(points.shape[:-1])
    for idx, sublattice in enumerate(phase_obj.constituents):
        vacancy_column = np.ones(points.shape[:-1])
        if 'VA' in set(sublattice):
            var_idx = variables.index(v.SiteFraction(phase_obj.name, idx, 'VA'))
            vacancy_column -= points[..., :, var_idx]
        site_ratio_normalization += phase_obj.sublattices[idx] * vacancy_column

    phase_compositions = np.empty(points.shape[:-1] + (len(components),))
    for col, comp in enumerate(components):
        avector = [float(vxx.species == comp) * \
            phase_obj.sublattices[vxx.sublattice_index] for vxx in variables]
        phase_compositions[..., :, col] = np.divide(np.dot(points[..., :, :], avector),
                                               site_ratio_normalization)

    coordinate_dict = {'component': components}
    coordinate_dict.update({key: np.atleast_1d(value) for key, value in statevar_dict.items()})
    output_columns = [str(x) for x in statevar_dict.keys()] + ['points']
    # Resize 'points' so it has the same number of columns as the maximum
    # number of internal degrees of freedom of any phase in the calculation.
    # We do this so that everything is aligned for concat.
    # Waste of memory? Yes, but the alternatives are unclear.
    expanded_points = np.full(points.shape[:-1] + (maximum_internal_dof,), np.nan)
    expanded_points[..., :points.shape[-1]] = points
    data_arrays = {'X': (output_columns + ['component'], phase_compositions),
                   'Phase': (output_columns,
                             np.full(points.shape[:-1], phase_obj.name, dtype='U'+str(len(phase_obj.name)))),
                   'Y': (output_columns + ['internal_dof'], expanded_points),
                   output: (output_columns, phase_output)
                   }

    return xray.Dataset(data_arrays, coords=coordinate_dict)


def calculate(dbf, comps, phases, mode=None, output='GM', fake_points=False, **kwargs):
    """
    Sample the property surface of 'output' containing the specified
    components and phases. Model parameters are taken from 'dbf' and any
    state variables (T, P, etc.) can be specified as keyword arguments.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : str or sequence
        Names of components to consider in the calculation.
    phases : str or sequence
        Names of phases to consider in the calculation.
    mode : string, optional
        See 'make_callable' docstring for details.
    output : string, optional
        Model attribute to sample.
    fake_points : bool, optional (Default: False)
        If True, the first few points of the output surface will be fictitious
        points used to define an equilibrium hyperplane guaranteed to be above
        all the other points. This is used for convex hull computations.
    points : ndarray or a dict of phase names to ndarray, optional
        Columns of ndarrays must be internal degrees of freedom, sorted.
        If this is not specified, points will be generated automatically.
    pdens : int, a dict of phase names to int, or a seq of both, optional
        Number of points to sample per degree of freedom.
    model : Model, a dict of phase names to Model, or a seq of both, optional
        Model class to use for each phase.

    Returns
    -------
    xray.Dataset of the sampled attribute as a function of state variables

    Examples
    --------
    None yet.
    """
    # Here we check for any keyword arguments that are special, i.e.,
    # there may be keyword arguments that aren't state variables
    pdens_dict = unpack_kwarg(kwargs.pop('pdens', 2000), default_arg=2000)
    points_dict = unpack_kwarg(kwargs.pop('points', None), default_arg=None)
    model_dict = unpack_kwarg(kwargs.pop('model', Model), default_arg=Model)
    callable_dict = unpack_kwarg(kwargs.pop('callables', None), default_arg=None)
    if isinstance(phases, str):
        phases = [phases]
    if isinstance(comps, str):
        comps = [comps]
    components = [x for x in sorted(comps) if not x.startswith('VA')]

    # Convert keyword strings to proper state variable objects
    # If we don't do this, sympy will get confused during substitution
    statevar_dict = collections.OrderedDict((v.StateVariable(key), unpack_condition(value)) \
                                            for (key, value) in sorted(kwargs.items()))
    str_statevar_dict = collections.OrderedDict((str(key), unpack_condition(value)) \
                                                for (key, value) in statevar_dict.items())
    all_phase_data = []
    comp_sets = {}
    largest_energy = -np.inf
    maximum_internal_dof = 0

    # Consider only the active phases
    active_phases = dict((name.upper(), dbf.phases[name.upper()]) \
        for name in unpack_phases(phases))

    for phase_name, phase_obj in sorted(active_phases.items()):
        # Build the symbolic representation of the energy
        mod = model_dict[phase_name]
        # if this is an object type, we need to construct it
        if isinstance(mod, type):
            try:
                model_dict[phase_name] = mod = mod(dbf, comps, phase_name)
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
            maximum_internal_dof = max(maximum_internal_dof, len(out.atoms(v.SiteFraction)))
        except AttributeError:
            raise AttributeError('Missing Model attribute {0} specified for {1}'
                                 .format(output, mod.__class__))

    for phase_name, phase_obj in sorted(active_phases.items()):
        try:
            mod = model_dict[phase_name]
        except KeyError:
            continue
        # Construct an ordered list of the variables
        variables, sublattice_dof = generate_dof(phase_obj, mod.components)
        out = getattr(mod, output)

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

        points = points_dict[phase_name]
        if points is None:
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
        # Ensure that points has the correct dimensions and dtype
        points = np.atleast_2d(np.asarray(points, dtype=np.float))

        phase_ds = _compute_phase_values(phase_obj, components, variables, str_statevar_dict,
                                         points, comp_sets[phase_name], output,
                                         maximum_internal_dof)
        # largest_energy is really only relevant if fake_points is set
        largest_energy = max(phase_ds[output].max(), largest_energy)
        all_phase_data.append(phase_ds)

    if fake_points:
        if output != 'GM':
            raise ValueError('fake_points=True should only be used with output=\'GM\'')
        phase_ds = _generate_fake_points(components, statevar_dict, largest_energy, output, maximum_internal_dof)
        final_ds = xray.concat(itertools.chain([phase_ds], all_phase_data),
                               dim='points')
    else:
        final_ds = xray.concat(all_phase_data, dim='points')

    # Reset the points dimension to use a single global index
    final_ds['points'] = np.arange(len(final_ds.points))
    return final_ds
