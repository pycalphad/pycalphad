"""
The calculate module contains a routine for calculating the
property surface of a system.
"""

import itertools
import warnings
from collections import OrderedDict
from collections.abc import Mapping
import numpy as np
from numpy import broadcast_to
import pycalphad.variables as v
from pycalphad import ConditionError
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.cache import cacheit
from pycalphad.core.light_dataset import LightDataset
from pycalphad.model import Model
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.utils import endmember_matrix, extract_parameters, \
    get_pure_elements, filter_phases, instantiate_models, point_sample, \
    unpack_components, unpack_condition, unpack_kwarg
from pycalphad.core.constants import MIN_SITE_FRACTION


def hr_point_sample(constraint_jac, constraint_rhs, initial_point, num_points):
    "Hit-and-run sampling of linearly-constrained site fraction spaces"
    q, r = np.linalg.qr(constraint_jac.T, mode='complete')
    q1 = q[:, :constraint_jac.shape[0]]
    q2 = q[:, constraint_jac.shape[0]:]
    r1 = r[:constraint_jac.shape[0], :]
    if initial_point is not None:
        z_bar = initial_point
    else:
        # minimum norm solution to underdetermined system of equations
        # may not be feasible if it fails the non-negativity constraint
        z_bar = np.linalg.lstsq(constraint_jac, constraint_rhs, rcond=None)[0]
    solution_norm = np.linalg.norm(constraint_jac.dot(z_bar) - constraint_rhs)
    if (solution_norm > 1e-4) or np.any(z_bar < 0):
        # initial point does not satisfy constraints; give up
        return np.empty((0, z_bar.shape[0]))
    # Hit-and-Run sampling
    new_feasible_z = np.zeros((num_points, constraint_jac.shape[1]))
    current_z = np.array(z_bar)
    min_z = MIN_SITE_FRACTION
    rng = np.random.RandomState(1769)
    for iteration in range(num_points):
        # generate unit direction in null space
        d = rng.normal(size=(constraint_jac.shape[1] - constraint_jac.shape[0]))
        d /= np.linalg.norm(d, axis=0)
        proj = np.dot(q2, d)
        # find extent of step direction possible while staying within bounds (0 <= z)
        with np.errstate(divide='ignore'):
            alphas = (min_z - current_z) / proj
        # Need to use small value to prevent constraints binding one sublattice (with proj ~ 0) from binding all dof
        max_alpha_candidates = alphas[np.logical_and(proj > 1e-6, np.isfinite(alphas))]
        min_alpha_candidates = alphas[np.logical_and(proj < -1e-6, np.isfinite(alphas))]
        alpha_min = np.min(min_alpha_candidates)
        alpha_max = np.max(max_alpha_candidates)
        # Poor progress; give up on sampling
        if np.abs(alpha_max - alpha_min) < 1e-4:
            new_feasible_z = new_feasible_z[:iteration, :]
            break
        # choose a random step size within the feasible interval
        new_alpha = rng.uniform(low=alpha_min, high=alpha_max)
        current_z += new_alpha * proj
        new_feasible_z[iteration, :] = current_z
    if np.any(new_feasible_z < 0):
        raise ValueError('Constrained sampling generated negative site fractions')
    return new_feasible_z


@cacheit
def _sample_phase_constitution(model, sampler, fixed_grid, pdens):
    """
    Sample the internal degrees of freedom of a phase.

    Parameters
    ----------
    model : Model
        Instance of a pycalphad Model
    sampler : Callable
        Callable returning an ArrayLike of points
    fixed_grid : bool
        If True, sample pdens points between each pair of endmembers
    pdens : int
        Number of points to sample in each sampled dimension

    Returns
    -------
    ndarray of points
    """
    # Eliminate pure vacancy endmembers from the calculation
    ALLOWED_CHARGE=1E-10
    vacancy_indices = []
    for sublattice in model.constituents:
        subl_va_indices = [idx for idx, spec in enumerate(sorted(set(sublattice))) if spec.number_of_atoms == 0]
        vacancy_indices.append(subl_va_indices)
    if len(vacancy_indices) != len(model.constituents):
        vacancy_indices = None
    sublattice_dof = [len(subl) for subl in model.constituents]
    # Add all endmembers to guarantee their presence
    points = endmember_matrix(sublattice_dof, vacancy_indices=vacancy_indices)
    site_ratios = model.site_ratios
    constant_site_ratios = True
    # The only implementation with variable site ratios is the two-sublattice ionic liquid.
    # This check is convenient for detecting 2SL ionic liquids without keeping other state.
    for sr in site_ratios:
        try:
            float(sr)
        except (TypeError, RuntimeError):
            constant_site_ratios = False
    species_charge = []
    for sublattice in range(len(model.constituents)):
        for species in sorted(model.constituents[sublattice]):
            species_charge.append(species.charge*site_ratios[sublattice])
    species_charge = np.array(species_charge)
    charge_constrained_space = constant_site_ratios and np.any(species_charge != 0)
    # We differentiate between (specifically) charge balance and general linear constraints for future use
    # This simplifies adding future constraints, such as disordered configuration sampling, or site fraction conditions
    # Note that if a phase only consists of site fraction balance constraints,
    # we do not consider that 'linearly constrained' for the purposes of sampling,
    # since the default sampler handles that case with an efficient method.
    linearly_constrained_space = charge_constrained_space

    if charge_constrained_space:
        endmembers = points
        Q = np.dot(endmembers, species_charge)
        # Sort endmembers by their charge
        charge_neutral_endmember_idxs = []
        charge_positive_endmember_idxs = []
        charge_negative_endmember_idxs = []
        for em_idx in range(endmembers.shape[0]):
            if Q[em_idx] > ALLOWED_CHARGE:
                charge_positive_endmember_idxs.append(em_idx)
            elif Q[em_idx] < -ALLOWED_CHARGE:
                charge_negative_endmember_idxs.append(em_idx)
            else:
                charge_neutral_endmember_idxs.append(em_idx)

        # Find all endmember pairs between the
        em_pts = [endmembers[em_idx] for em_idx in charge_neutral_endmember_idxs]
        for pos_em_idx, neg_em_idx in itertools.product(charge_positive_endmember_idxs, charge_negative_endmember_idxs):
            # Solve equation: Q_{pos}*x + Q_{neg}(1-x) = 0
            x = - Q[neg_em_idx] / (Q[pos_em_idx] - Q[neg_em_idx])
            em_pts.append(endmembers[pos_em_idx] * x + endmembers[neg_em_idx] * (1-x))

        # Charge neutral endmembers and mixed pseudo-endmembers
        points = np.asarray(em_pts)

    if (fixed_grid is True) and not linearly_constrained_space:
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
        if linearly_constrained_space:
            # construct constraint Jacobian for this phase
            # Model technically already does this so it would be better to reuse that functionality
            # number of sublattices, plus charge balance
            num_constraints = len(sublattice_dof) + 1
            constraint_jac = np.zeros((num_constraints, points.shape[-1]))
            constraint_rhs = np.zeros(num_constraints)
            # site fraction balance
            dof_idx = 0
            constraint_idx = 0
            for subl_dof in sublattice_dof:
                constraint_jac[constraint_idx, dof_idx:dof_idx + subl_dof] = 1
                constraint_rhs[constraint_idx] = 1
                constraint_idx += 1
                dof_idx += subl_dof
            # charge balance
            constraint_jac[constraint_idx, :] = species_charge
            constraint_rhs[constraint_idx] = 0
            # Sample additional points which obey the constraints
            # Mean of pseudo-endmembers is feasible by convexity of the space
            initial_point = np.mean(points, axis=0)
            num_points = (pdens ** 2) * (constraint_jac.shape[1] - constraint_jac.shape[0])
            extra_points = hr_point_sample(constraint_jac, constraint_rhs, initial_point, num_points)
            points = np.concatenate((points, extra_points))
            assert np.max(np.abs(constraint_jac.dot(points.T).T - constraint_rhs)) < 1e-6
            if points.shape[0] == 0:
                warnings.warn(f'{model.phase_name} has zero feasible configurations under the given conditions')
        else:
            points = np.concatenate((points, sampler(sublattice_dof, pdof=pdens)))

    # Filter out nan's that may have slipped in if we sampled too high a vacancy concentration
    # Issues with this appear to be platform-dependent
    points = points[~np.isnan(points).any(axis=-1)]
    # Ensure that points has the correct dimensions and dtype
    points = np.atleast_2d(np.asarray(points, dtype=np.float_))
    return points


def _compute_phase_values(components, statevar_dict,
                          points, phase_record, output, maximum_internal_dof, broadcast=True,
                          parameters=None, fake_points=False,
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
    parameters : OrderedDict {str -> float or sequence}, optional
        Maps SymEngine symbols to a scalar or 1-D array. The arrays must be equal length.
        The corresponding PhaseRecord must have been initialized with the same parameters.
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
    pure_elements = [x for x in pure_elements if x != 'VA']
    # func may only have support for vectorization along a single axis (no broadcasting)
    # we need to force broadcasting and flatten the result before calling
    bc_statevars = np.ascontiguousarray([broadcast_to(x, points.shape[:-1]).reshape(-1) for x in statevars])
    pts = points.reshape(-1, points.shape[-1])
    dof = np.ascontiguousarray(np.concatenate((bc_statevars.T, pts), axis=1))
    phase_compositions = np.zeros((dof.shape[0], len(pure_elements)), order='F')

    param_symbols, parameter_array = extract_parameters(parameters)
    parameter_array_length = parameter_array.shape[0]
    if parameter_array_length == 0:
        # No parameters specified
        phase_output = np.zeros(dof.shape[0], order='C')
        phase_record.obj_2d(phase_output, dof)
    else:
        # Vectorized parameter arrays
        phase_output = np.zeros((dof.shape[0], parameter_array_length), order='C')
        phase_record.obj_parameters_2d(phase_output, dof, parameter_array)

    for el_idx in range(len(pure_elements)):
        phase_record.mass_obj_2d(phase_compositions[:, el_idx], dof, el_idx)

    max_tieline_vertices = len(pure_elements)
    if isinstance(phase_output, (float, int)):
        phase_output = broadcast_to(phase_output, points.shape[:-1])
    if isinstance(phase_compositions, (float, int)):
        phase_compositions = broadcast_to(phase_output, points.shape[:-1] + (len(pure_elements),))
    phase_output = np.asarray(phase_output, dtype=np.float_)
    if parameter_array_length <= 1:
        phase_output.shape = points.shape[:-1]
    else:
        phase_output.shape = points.shape[:-1] + (parameter_array_length,)
    phase_compositions = np.asarray(phase_compositions, dtype=np.float_)
    phase_compositions.shape = points.shape[:-1] + (len(pure_elements),)
    if fake_points:
        output_shape = points.shape[:-2] + (max_tieline_vertices,)
        if parameter_array_length > 1:
            output_shape = output_shape + (parameter_array_length,)
            concat_axis = -2
        else:
            concat_axis = -1
        phase_output = np.concatenate((broadcast_to(largest_energy, output_shape), phase_output), axis=concat_axis)
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
    # In each case, first check if we need to do this...
    # It can be expensive for many points (~14s for 500M points)
    if fake_points:
        desired_shape = points.shape[:-2] + (max_tieline_vertices + points.shape[-2], maximum_internal_dof)
        expanded_points = np.full(desired_shape, np.nan)
        expanded_points[..., len(pure_elements):, :points.shape[-1]] = points
    else:
        desired_shape = points.shape[:-1] + (maximum_internal_dof,)
        if points.shape == desired_shape:
            expanded_points = points
        else:
            # TODO: most optimal solution would be to take pre-extended arrays as an argument and remove this
            # This still copies the array, but is more efficient than filling
            # an array with np.nan, then copying the existing points
            append_nans = np.full(desired_shape[:-1] + (desired_shape[-1] - points.shape[-1],), np.nan)
            expanded_points = np.append(points, append_nans, axis=-1)
    if broadcast:
        coordinate_dict.update({key: np.atleast_1d(value) for key, value in statevar_dict.items()})
        output_columns = [str(x) for x in statevar_dict.keys()] + ['points']
    else:
        output_columns = ['points']
    if parameter_array_length > 1:
        parameter_column = ['samples']
        coordinate_dict['param_symbols'] = [str(x) for x in param_symbols]
    else:
        parameter_column = []
    data_arrays = {'X': (output_columns + ['component'], np.ascontiguousarray(phase_compositions)),
                   'Phase': (output_columns, phase_names),
                   'Y': (output_columns + ['internal_dof'], expanded_points),
                   output: (['dim_'+str(i) for i in range(len(phase_output.shape) - (len(output_columns)+len(parameter_column)))] + output_columns + parameter_column, phase_output)
                   }
    if not broadcast:
        # Add state variables as data variables rather than as coordinates
        for sym, vals in zip(statevar_dict.keys(), statevars):
            data_arrays.update({sym: (output_columns, vals)})
    if parameter_array_length > 1:
        data_arrays['param_values'] = (['samples', 'param_symbols'], parameter_array)
    return LightDataset(data_arrays, coords=coordinate_dict)


def calculate(dbf, comps, phases, mode=None, output='GM', fake_points=False, broadcast=True, parameters=None, to_xarray=True, phase_records=None, **kwargs):
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
        Maps SymEngine Symbol to numbers, for overriding the values of parameters in the Database.
    phase_records : Optional[Mapping[str, PhaseRecord]]
        Mapping of phase names to PhaseRecord objects. Must include all active phases.
        The `model` argument must be a mapping of phase names to instances of Model
        objects. Callers must take care that the PhaseRecord objects were created with
        the same `output` as passed to `calculate`.

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
    callables = kwargs.pop('callables', {})
    sampler_dict = unpack_kwarg(kwargs.pop('sampler', None), default_arg=None)
    fixedgrid_dict = unpack_kwarg(kwargs.pop('grid_points', True), default_arg=True)
    model = kwargs.pop('model', None)
    parameters = parameters or dict()
    if isinstance(parameters, dict):
        parameters = OrderedDict(sorted(parameters.items(), key=str))
    if isinstance(phases, str):
        phases = [phases]
    if isinstance(comps, (str, v.Species)):
        comps = [comps]
    comps = sorted(unpack_components(dbf, comps))
    if points_dict is None and broadcast is False:
        raise ValueError('The \'points\' keyword argument must be specified if broadcast=False is also given.')
    nonvacant_components = [x for x in sorted(comps) if x.number_of_atoms > 0]
    nonvacant_elements = get_pure_elements(dbf, comps)

    all_phase_data = []
    largest_energy = 1e10

    # Consider only the active phases
    list_of_possible_phases = filter_phases(dbf, comps)
    if len(list_of_possible_phases) == 0:
        raise ConditionError('There are no phases in the Database that can be active with components {0}'.format(comps))
    active_phases = filter_phases(dbf, comps, phases)
    if len(active_phases) == 0:
        raise ConditionError('None of the passed phases ({0}) are active. List of possible phases: {1}.'.format(phases, list_of_possible_phases))

    if isinstance(output, (list, tuple, set)):
        raise NotImplementedError('Only one property can be specified in calculate() at a time')
    output = output if output is not None else 'GM'

    # Implicitly add 'N' state variable as a string to keyword arguements if it's not passed
    if kwargs.get('N') is None:
        kwargs['N'] = 1
    if np.any(np.array(kwargs['N']) != 1):
        raise ConditionError('N!=1 is not yet supported, got N={}'.format(kwargs['N']))

    # TODO: conditions dict of StateVariable instances should become part of the calculate API
    statevar_strings = [sv for sv in kwargs.keys() if getattr(v, sv) is not None]
    # If we don't do this, sympy will get confused during substitution
    statevar_dict = dict((v.StateVariable(key), unpack_condition(value)) for key, value in kwargs.items() if key in statevar_strings)
    # Sort after default state variable check to fix gh-116
    statevar_dict = OrderedDict(sorted(statevar_dict.items(), key=lambda x: str(x[0])))
    str_statevar_dict = OrderedDict((str(key), unpack_condition(value)) for (key, value) in statevar_dict.items())

    # Build phase records if they weren't passed
    if phase_records is None:
        models = instantiate_models(dbf, comps, active_phases, model=model, parameters=parameters)
        phase_records = build_phase_records(dbf, comps, active_phases, statevar_dict,
                                            models=models, parameters=parameters,
                                            output=output, callables=callables,
                                            build_gradients=False, build_hessians=False,
                                            verbose=kwargs.pop('verbose', False))
    else:
        # phase_records were provided, instantiated models must also be provided by the caller
        models = model
        if not isinstance(models, Mapping):
            raise ValueError("A dictionary of instantiated models must be passed to `equilibrium` with the `model` argument if the `phase_records` argument is used.")
        active_phases_without_models = [name for name in active_phases if not isinstance(models.get(name), Model)]
        active_phases_without_phase_records = [name for name in active_phases if not isinstance(phase_records.get(name), PhaseRecord)]
        if len(active_phases_without_phase_records) > 0:
            raise ValueError(f"phase_records must contain a PhaseRecord instance for every active phase. Missing PhaseRecord objects for {sorted(active_phases_without_phase_records)}")
        if len(active_phases_without_models) > 0:
            raise ValueError(f"model must contain a Model instance for every active phase. Missing Model objects for {sorted(active_phases_without_models)}")

    maximum_internal_dof = max(len(models[phase_name].site_fractions) for phase_name in active_phases)
    for phase_name in sorted(active_phases):
        mod = models[phase_name]
        phase_record = phase_records[phase_name]
        points = points_dict[phase_name]
        if points is None:
            points = _sample_phase_constitution(mod, sampler_dict[phase_name] or point_sample,
                                                fixedgrid_dict[phase_name], pdens_dict[phase_name])
        points = np.atleast_2d(points)

        fp = fake_points and (phase_name == sorted(active_phases)[0])
        phase_ds = _compute_phase_values(nonvacant_components, str_statevar_dict,
                                         points, phase_record, output,
                                         maximum_internal_dof, broadcast=broadcast, parameters=parameters,
                                         largest_energy=float(largest_energy), fake_points=fp)
        all_phase_data.append(phase_ds)

    fp_offset = len(nonvacant_elements) if fake_points else 0
    running_total = [fp_offset] + list(np.cumsum([phase_ds['X'].shape[-2] for phase_ds in all_phase_data]))
    islice_by_phase = {phase_name: slice(running_total[phase_idx], running_total[phase_idx+1], None)
                       for phase_idx, phase_name in enumerate(sorted(active_phases))}
    # speedup for single-phase case (found by profiling)
    if len(all_phase_data) > 1:
        concatenated_coords = all_phase_data[0].coords

        data_vars = all_phase_data[0].data_vars
        concatenated_data_vars = {}
        for var in data_vars.keys():
            data_coords = data_vars[var][0]
            points_idx = data_coords.index('points')  # concatenation axis
            arrs = []
            for phase_data in all_phase_data:
                arrs.append(getattr(phase_data, var))
            concat_data = np.concatenate(arrs, axis=points_idx)
            concatenated_data_vars[var] = (data_coords, concat_data)
        final_ds = LightDataset(data_vars=concatenated_data_vars, coords=concatenated_coords)
    else:
        final_ds = all_phase_data[0]
    final_ds.attrs['phase_indices'] = islice_by_phase
    if to_xarray:
        return final_ds.get_dataset()
    else:
        return final_ds
