"""
The calculate module contains a routine for calculating the
property surface of a system.
"""

from __future__ import division
from pycalphad import Model
from pycalphad.model import DofError
from pycalphad.core.sympydiff_utils import build_functions
from pycalphad.core.utils import point_sample, generate_dof
from pycalphad.core.utils import endmember_matrix, unpack_kwarg
from pycalphad.core.utils import broadcast_to, unpack_condition, unpack_phases, unpack_components
from pycalphad.core.cache import cacheit
from pycalphad.core.phase_rec import PhaseRecord, PhaseRecord_from_cython
import pycalphad.variables as v
from sympy import Symbol
import numpy as np
import itertools
import collections
import warnings
from xarray import Dataset, concat
from collections import OrderedDict


def _generate_fake_points(components, statevar_dict, energy_limit, output, maximum_internal_dof, broadcast):
    """
    Generate points for a fictitious hyperplane used as a starting point for energy minimization.
    """
    coordinate_dict = {'component': components}
    largest_energy = float(energy_limit)
    if largest_energy < 0:
        largest_energy *= 0.01
    else:
        largest_energy *= 10
    if broadcast:
        output_columns = [str(x) for x in statevar_dict.keys()] + ['points']
        statevar_shape = tuple(len(np.atleast_1d(x)) for x in statevar_dict.values())
        coordinate_dict.update({str(key): value for key, value in statevar_dict.items()})
        # The internal dof for the fake points are all NaNs
        expanded_points = np.full(statevar_shape + (len(components), maximum_internal_dof), np.nan)
        data_arrays = {'X': (output_columns + ['component'],
                             broadcast_to(np.eye(len(components)), statevar_shape + (len(components), len(components)))),
                       'Y': (output_columns + ['internal_dof'], expanded_points),
                       'Phase': (output_columns, np.full(statevar_shape + (len(components),), '_FAKE_', dtype='S6')),
                       output: (output_columns, np.full(statevar_shape + (len(components),), largest_energy))
                       }
    else:
        output_columns = ['points']
        statevar_shape = (len(components) * max([len(np.atleast_1d(x)) for x in statevar_dict.values()]),)
        # The internal dof for the fake points are all NaNs
        expanded_points = np.full(statevar_shape + (maximum_internal_dof,), np.nan)
        data_arrays = {'X': (output_columns + ['component'],
                             broadcast_to(np.tile(np.eye(len(components)), (statevar_shape[0] / len(components), 1)),
                                                  statevar_shape + (len(components),))),
                       'Y': (output_columns + ['internal_dof'], expanded_points),
                       'Phase': (output_columns, np.full(statevar_shape, '_FAKE_', dtype='S6')),
                       output: (output_columns, np.full(statevar_shape, largest_energy))
                       }
        # Add state variables as data variables if broadcast=False
        data_arrays.update({str(key): (output_columns, np.repeat(value, len(components)))
                            for key, value in statevar_dict.items()})
    return Dataset(data_arrays, coords=coordinate_dict)


@cacheit
def _sample_phase_constitution(phase_name, phase_constituents, sublattice_dof, comps,
                               variables, sampler, fixed_grid, pdens):
    """
    Sample the internal degrees of freedom of a phase.

    Parameters
    ----------
    phase_name
    phase_constituents
    sublattice_dof
    comps
    variables
    sampler
    fixed_grid
    pdens

    Returns
    -------
    ndarray of points
    """
    # Eliminate pure vacancy endmembers from the calculation
    vacancy_indices = list()
    for idx, sublattice in enumerate(phase_constituents):
        active_in_subl = sorted(set(phase_constituents[idx]).intersection(comps))
        is_vacancy = [spec.number_of_atoms == 0 for spec in active_in_subl]
        subl_va_indices = list(idx for idx, x in enumerate(is_vacancy) if x == True)
        vacancy_indices.append(subl_va_indices)
    if len(vacancy_indices) != len(phase_constituents):
        vacancy_indices = None
    # Add all endmembers to guarantee their presence
    points = endmember_matrix(sublattice_dof,
                              vacancy_indices=vacancy_indices)
    if fixed_grid is True:
        # Sample along the edges of the endmembers
        # These constitution space edges are often the equilibrium points!
        em_pairs = list(itertools.combinations(points, 2))
        lingrid = np.linspace(0, 1, pdens)
        extra_points = [first_em * lingrid[np.newaxis].T +
                        second_em * lingrid[::-1][np.newaxis].T
                        for first_em, second_em in em_pairs]
        points = np.concatenate(list(itertools.chain([points], extra_points)))

    # Sample composition space for more points
    if sum(sublattice_dof) > len(sublattice_dof):
        points = np.concatenate((points,
                                 sampler(sublattice_dof,
                                         pdof=pdens)
                                 ))
    # Filter out nan's that may have slipped in if we sampled too high a vacancy concentration
    # Issues with this appear to be platform-dependent
    points = points[~np.isnan(points).any(axis=-1)]
    # Ensure that points has the correct dimensions and dtype
    points = np.atleast_2d(np.asarray(points, dtype=np.float))
    return points


def _compute_phase_values(components, statevar_dict,
                          points, phase_record, output, maximum_internal_dof, broadcast=True, fake_points=False,
                          largest_energy=None):
    """
    Calculate output values for a particular phase.

    Parameters
    ----------
    components : list
        Names of components to consider in the calculation.
    statevar_dict : OrderedDict {str -> float or sequence}
        Mapping of state variables to desired values. This will broadcast if necessary.
    points : ndarray
        Inputs to 'func', except state variables. Columns should be in 'variables' order.
    phase_record : PhaseRecord
        Contains callable for energy and phase metadata.
    output : string
        Desired name of the output result in the Dataset.
    maximum_internal_dof : int
        Largest number of internal degrees of freedom of any phase. This is used
        to guarantee different phase's Datasets can be concatenated.
    broadcast : bool
        If True, broadcast state variables against each other to create a grid.
        If False, assume state variables are given as equal-length lists (or single-valued).
    fake_points : bool, optional (Default: False)
        If True, the first few points of the output surface will be fictitious
        points used to define an equilibrium hyperplane guaranteed to be above
        all the other points. This is used for convex hull computations.

    Returns
    -------
    Dataset of the output attribute as a function of state variables

    Examples
    --------
    None yet.
    """
    if broadcast:
        # Broadcast compositions and state variables along orthogonal axes
        # This lets us eliminate an expensive Python loop
        statevars = np.meshgrid(*itertools.chain(statevar_dict.values(),
                                                     [np.empty(points.shape[-2])]),
                                    sparse=True, indexing='ij')[:-1]
        points = broadcast_to(points, tuple(len(np.atleast_1d(x)) for x in statevar_dict.values()) + points.shape[-2:])
    else:
        statevars = list(np.atleast_1d(x) for x in statevar_dict.values())
        statevars_ = []
        for statevar in statevars:
            if (len(statevar) != len(points)) and (len(statevar) == 1):
                statevar = np.repeat(statevar, len(points))
            if (len(statevar) != len(points)) and (len(statevar) != 1):
                raise ValueError('Length of state variable list and number of given points must be equal when '
                                 'broadcast=False.')
            statevars_.append(statevar)
        statevars = statevars_
    pure_elements = [list(x.constituents.keys()) for x in components]
    pure_elements = sorted(set([el.upper() for constituents in pure_elements for el in constituents]))
    # func may only have support for vectorization along a single axis (no broadcasting)
    # we need to force broadcasting and flatten the result before calling
    bc_statevars = [np.ascontiguousarray(broadcast_to(x, points.shape[:-1]).reshape(-1)) for x in statevars]
    pts = points.reshape(-1, points.shape[-1]).T
    dof = np.ascontiguousarray(np.concatenate((bc_statevars, pts), axis=0).T)
    phase_output = np.ascontiguousarray(np.zeros(dof.shape[0]))
    phase_compositions = np.asfortranarray(np.zeros((dof.shape[0], len(pure_elements))))
    phase_record.obj(phase_output, dof)
    for el_idx in range(len(pure_elements)):
        phase_record.mass_obj(phase_compositions[:,el_idx], dof, el_idx)

    max_tieline_vertices = len(pure_elements)
    if isinstance(phase_output, (float, int)):
        phase_output = broadcast_to(phase_output, points.shape[:-1])
    if isinstance(phase_compositions, (float, int)):
        phase_compositions = broadcast_to(phase_output, points.shape[:-1] + (len(pure_elements),))
    phase_output = np.asarray(phase_output, dtype=np.float)
    phase_output.shape = points.shape[:-1]
    phase_compositions = np.asarray(phase_compositions, dtype=np.float)
    phase_compositions.shape = points.shape[:-1] + (len(pure_elements),)
    if fake_points:
        phase_output = np.concatenate((broadcast_to(largest_energy, points.shape[:-2] + (max_tieline_vertices,)), phase_output), axis=-1)
        phase_names = np.concatenate((broadcast_to('_FAKE_', points.shape[:-2] + (max_tieline_vertices,)),
                                      np.full(points.shape[:-1], phase_record.phase_name, dtype='U' + str(len(phase_record.phase_name)))), axis=-1)
    else:
        phase_names = np.full(points.shape[:-1], phase_record.phase_name, dtype='U'+str(len(phase_record.phase_name)))

    if fake_points:
        phase_compositions = np.concatenate((np.broadcast_to(np.eye(len(pure_elements)), points.shape[:-2] + (max_tieline_vertices, len(pure_elements))), phase_compositions), axis=-2)

    coordinate_dict = {'component': pure_elements}
    # Resize 'points' so it has the same number of columns as the maximum
    # number of internal degrees of freedom of any phase in the calculation.
    # We do this so that everything is aligned for concat.
    # Waste of memory? Yes, but the alternatives are unclear.
    if fake_points:
        expanded_points = np.full(points.shape[:-2] + (max_tieline_vertices + points.shape[-2], maximum_internal_dof), np.nan)
        expanded_points[..., len(pure_elements):, :points.shape[-1]] = points
    else:
        expanded_points = np.full(points.shape[:-1] + (maximum_internal_dof,), np.nan)
        expanded_points[..., :points.shape[-1]] = points
    if broadcast:
        coordinate_dict.update({key: np.atleast_1d(value) for key, value in statevar_dict.items()})
        output_columns = [str(x) for x in statevar_dict.keys()] + ['points']
    else:
        output_columns = ['points']
    data_arrays = {'X': (output_columns + ['component'], phase_compositions),
                   'Phase': (output_columns, phase_names),
                   'Y': (output_columns + ['internal_dof'], expanded_points),
                   output: (['dim_'+str(i) for i in range(len(phase_output.shape) - len(output_columns))] + output_columns, phase_output)
                   }
    if not broadcast:
        # Add state variables as data variables rather than as coordinates
        for sym, vals in zip(statevar_dict.keys(), statevars):
            data_arrays.update({sym: (output_columns, vals)})

    return Dataset(data_arrays, coords=coordinate_dict)


def calculate(dbf, comps, phases, mode=None, output='GM', fake_points=False, broadcast=True, parameters=None, **kwargs):
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
    broadcast : bool, optional
        If True, broadcast given state variable lists against each other to create a grid.
        If False, assume state variables are given as equal-length lists.
    points : ndarray or a dict of phase names to ndarray, optional
        Columns of ndarrays must be internal degrees of freedom (site fractions), sorted.
        If this is not specified, points will be generated automatically.
    pdens : int, a dict of phase names to int, or a seq of both, optional
        Number of points to sample per degree of freedom.
        Default: 2000; Default when called from equilibrium(): 500
    model : Model, a dict of phase names to Model, or a seq of both, optional
        Model class to use for each phase.
    sampler : callable, a dict of phase names to callable, or a seq of both, optional
        Function to sample phase constitution space.
        Must have same signature as 'pycalphad.core.utils.point_sample'
    grid_points : bool, a dict of phase names to bool, or a seq of both, optional (Default: True)
        Whether to add evenly spaced points between end-members.
        The density of points is determined by 'pdens'
    parameters : dict, optional
        Maps SymPy Symbol to numbers, for overriding the values of parameters in the Database.

    Returns
    -------
    Dataset of the sampled attribute as a function of state variables

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
    mass_dict = unpack_kwarg(kwargs.pop('massfuncs', None), default_arg=None)
    sampler_dict = unpack_kwarg(kwargs.pop('sampler', None), default_arg=None)
    fixedgrid_dict = unpack_kwarg(kwargs.pop('grid_points', True), default_arg=True)
    parameters = parameters or dict()
    if isinstance(parameters, dict):
        parameters = OrderedDict(sorted(parameters.items(), key=str))
    param_symbols = tuple(parameters.keys())
    param_values = np.atleast_1d(np.array(list(parameters.values()), dtype=np.float))
    if isinstance(phases, str):
        phases = [phases]
    if isinstance(comps, (str, v.Species)):
        comps = [comps]
    comps = sorted(unpack_components(dbf, comps))
    if points_dict is None and broadcast is False:
        raise ValueError('The \'points\' keyword argument must be specified if broadcast=False is also given.')
    nonvacant_components = [x for x in sorted(comps) if x.number_of_atoms > 0]

    # Convert keyword strings to proper state variable objects
    # If we don't do this, sympy will get confused during substitution
    statevar_dict = dict((v.StateVariable(key), unpack_condition(value)) for (key, value) in kwargs.items())
    # XXX: CompiledModel assumes P, T are the only state variables
    if statevar_dict.get(v.P, None) is None:
        statevar_dict[v.P] = 101325
    if statevar_dict.get(v.T, None) is None:
        statevar_dict[v.T] = 300
    # Sort after default state variable check to fix gh-116
    statevar_dict = collections.OrderedDict(sorted(statevar_dict.items(), key=lambda x: str(x[0])))
    str_statevar_dict = collections.OrderedDict((str(key), unpack_condition(value)) \
                                                for (key, value) in statevar_dict.items())
    all_phase_data = []
    comp_sets = {}
    largest_energy = 1e30
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
                model_dict[phase_name] = mod = mod(dbf, comps, phase_name, parameters=parameters)
            except DofError:
                # we can't build the specified phase because the
                # specified components aren't found in every sublattice
                # we'll just skip it
                warnings.warn("""Suspending specified phase {} due to
                some sublattices containing only unspecified components""".format(phase_name))
                continue
        if points_dict[phase_name] is None:
            maximum_internal_dof = max(maximum_internal_dof, sum(len(x) for x in mod.constituents))
        else:
            maximum_internal_dof = max(maximum_internal_dof, np.asarray(points_dict[phase_name]).shape[-1])

    for phase_name, phase_obj in sorted(active_phases.items()):
        try:
            mod = model_dict[phase_name]
        except KeyError:
            continue
        # this is a phase model we couldn't construct for whatever reason; skip it
        if isinstance(mod, type):
            continue
        # Construct an ordered list of the variables
        variables, sublattice_dof = generate_dof(phase_obj, mod.components)
        # Build the "fast" representation of that model
        if callable_dict[phase_name] is None:
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
                warnings.warn('Setting undefined symbol {0} for phase {1} to zero'.format(undef, phase_name))
            comp_sets[phase_name] = build_functions(out, list(statevar_dict.keys()) + variables,
                                                    include_obj=True, include_grad=False,
                                                    parameters=param_symbols)
        else:
            comp_sets[phase_name] = callable_dict[phase_name]
        if mass_dict[phase_name] is None:
            pure_elements = [spec for spec in nonvacant_components
                             if (len(spec.constituents.keys()) == 1 and
                                 list(spec.constituents.keys())[0] == spec.name)
                             ]
            # TODO: In principle, we should also check for undefs in mod.moles()
            mass_dict[phase_name] = [build_functions(mod.moles(el), list(statevar_dict.keys()) + variables,
                                                     include_obj=True, include_grad=False,
                                                     parameters=param_symbols)
                                     for el in pure_elements]
        phase_record = PhaseRecord_from_cython(comps, list(statevar_dict.keys()) + variables,
                                    np.array(dbf.phases[phase_name].sublattices, dtype=np.float),
                                    param_values, comp_sets[phase_name], None, None, mass_dict[phase_name], None)
        points = points_dict[phase_name]
        if points is None:
            points = _sample_phase_constitution(phase_name, phase_obj.constituents, sublattice_dof, comps,
                                                tuple(variables), sampler_dict[phase_name] or point_sample,
                                                fixedgrid_dict[phase_name], pdens_dict[phase_name])
        points = np.atleast_2d(points)

        fp = fake_points and (phase_name == sorted(active_phases.keys())[0])
        phase_ds = _compute_phase_values(nonvacant_components, str_statevar_dict,
                                         points, phase_record, output,
                                         maximum_internal_dof, broadcast=broadcast,
                                         largest_energy=float(largest_energy), fake_points=fp)
        all_phase_data.append(phase_ds)

    # speedup for single-phase case (found by profiling)
    if len(all_phase_data) > 1:
        final_ds = concat(all_phase_data, dim='points')
        final_ds['points'].values = np.arange(len(final_ds['points']))
        final_ds.coords['points'].values = np.arange(len(final_ds['points']))
    else:
        final_ds = all_phase_data[0]
    return final_ds
