"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""
from __future__ import print_function
import warnings
import pycalphad.variables as v
from pycalphad.core.utils import unpack_components, unpack_condition, unpack_phases, filter_phases, instantiate_models, get_state_variables
from pycalphad import calculate
from pycalphad.core.errors import EquilibriumError, ConditionError
from pycalphad.core.starting_point import starting_point
from pycalphad.codegen.callables import build_phase_records
from pycalphad.core.constants import MIN_SITE_FRACTION
from pycalphad.core.eqsolver import _solve_eq_at_conditions
from pycalphad.core.solver import InteriorPointSolver
import dask
from dask import delayed
from xarray import Dataset
import numpy as np
from collections import OrderedDict
from datetime import datetime


def _adjust_conditions(conds):
    "Adjust conditions values to be within the numerical limit of the solver."
    new_conds = OrderedDict()
    for key, value in sorted(conds.items(), key=str):
        if key == str(key):
            key = getattr(v, key, key)
        if isinstance(key, v.Composition):
            new_conds[key] = [max(val, MIN_SITE_FRACTION*1000) for val in unpack_condition(value)]
        else:
            new_conds[key] = unpack_condition(value)
    return new_conds


def _merge_property_slices(properties, chunk_grid, slices, conds_keys, results):
    "Merge back together slices of 'properties'."
    for prop_slice, prop_arr in zip(chunk_grid, results):
        if not isinstance(prop_arr, Dataset):
            print('Error: {}'.format(prop_arr))
            continue
        all_coords = dict(zip(conds_keys, [np.atleast_1d(sl)[ch]
                                                               for ch, sl in zip(prop_slice, slices)]))
        for dv in properties.data_vars.keys():
            # Have to be very careful with how we assign to 'properties' here
            # We may accidentally assign to a copy unless we index the data variable first
            dv_coords = {key: val for key, val in all_coords.items() if key in properties[dv].coords.keys()}
            properties[dv][dv_coords] = prop_arr[dv]
    return properties


def _eqcalculate(dbf, comps, phases, conditions, output, data=None, per_phase=False, callables=None, parameters=None,
                 **kwargs):
    """
    WARNING: API/calling convention not finalized.
    Compute the *equilibrium value* of a property.
    This function differs from `calculate` in that it computes
    thermodynamic equilibrium instead of randomly sampling the
    internal degrees of freedom of a phase.
    Because of that, it's slower than `calculate`.
    This plugs in the equilibrium phase and site fractions
    to compute a thermodynamic property defined in a Model.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list or dict
        Names of phases to consider in the calculation.
    conditions : dict or (list of dict)
        StateVariables and their corresponding value.
    output : str
        Equilibrium model property (e.g., CPM, HM, etc.) to compute.
        This must be defined as an attribute in the Model class of each phase.
    data : Dataset, optional
        Previous result of call to `equilibrium`.
        Should contain the equilibrium configurations at the conditions of interest.
        If the databases are not the same as in the original calculation,
        the results may be meaningless. If None, `equilibrium` will be called.
        Specifying this keyword argument can save the user some time if several properties
        need to be calculated in succession.
    per_phase : bool, optional
        If True, compute and return the property for each phase present.
        If False, return the total system value, weighted by the phase fractions.
    parameters : dict, optional
        Maps SymPy Symbol to numbers, for overriding the values of parameters in the Database.
    callables : dict
        Callable functions to compute 'output' for each phase.
    kwargs
        Passed to `calculate`.

    Returns
    -------
    Dataset of property as a function of equilibrium conditions
    """
    if data is None:
        data = equilibrium(dbf, comps, phases, conditions)
    active_phases = unpack_phases(phases) or sorted(dbf.phases.keys())
    conds = _adjust_conditions(conditions)
    indep_vars = ['N', 'P', 'T']
    # TODO: Rewrite this to use the coord dict from 'data'
    str_conds = OrderedDict((str(key), value) for key, value in conds.items())
    indep_vals = list([float(x) for x in np.atleast_1d(val)]
                      for key, val in str_conds.items() if key in indep_vars)
    coord_dict = str_conds.copy()
    components = [x for x in sorted(comps)]
    desired_active_pure_elements = [list(x.constituents.keys()) for x in components]
    desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements for el in constituents]
    pure_elements = sorted(set([x for x in desired_active_pure_elements if x != 'VA']))
    coord_dict['vertex'] = np.arange(len(pure_elements) + 1)  # +1 is to accommodate the degenerate degree of freedom at the invariant reactions
    grid_shape = np.meshgrid(*coord_dict.values(),
                             indexing='ij', sparse=False)[0].shape
    prop_shape = grid_shape
    prop_dims = list(str_conds.keys()) + ['vertex']

    result = Dataset({output: (prop_dims, np.full(prop_shape, np.nan))}, coords=coord_dict)
    # For each phase select all conditions where that phase exists
    # Perform the appropriate calculation and then write the result back
    for phase in active_phases:
        dof = sum([len(x) for x in dbf.phases[phase].constituents])
        current_phase_indices = (data.Phase.values == phase)
        if ~np.any(current_phase_indices):
            continue
        points = data.Y.values[np.nonzero(current_phase_indices)][..., :dof]
        statevar_indices = np.nonzero(current_phase_indices)[:len(indep_vals)]
        statevars = {key: np.take(np.asarray(vals), idx)
                     for key, vals, idx in zip(indep_vars, indep_vals, statevar_indices)}
        statevars.update(kwargs)
        if statevars.get('mode', None) is None:
            statevars['mode'] = 'numpy'
        calcres = calculate(dbf, comps, [phase], output=output, points=points, broadcast=False,
                            callables=callables, parameters=parameters, **statevars)
        result[output].values[np.nonzero(current_phase_indices)] = calcres[output].values
    if not per_phase:
        result[output] = (result[output] * data['NP']).sum(dim='vertex', skipna=True)
    else:
        result['Phase'] = data['Phase'].copy()
        result['NP'] = data['NP'].copy()
    return result


def equilibrium(dbf, comps, phases, conditions, output=None, model=None,
                verbose=False, broadcast=True, calc_opts=None,
                scheduler='sync', parameters=None, solver=None, callables=None,
                **kwargs):
    """
    Calculate the equilibrium state of a system containing the specified
    components and phases, under the specified conditions.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list or dict
        Names of phases to consider in the calculation.
    conditions : dict or (list of dict)
        StateVariables and their corresponding value.
    output : str or list of str, optional
        Additional equilibrium model properties (e.g., CPM, HM, etc.) to compute.
        These must be defined as attributes in the Model class of each phase.
    model : Model, a dict of phase names to Model, or a seq of both, optional
        Model class to use for each phase.
    verbose : bool, optional
        Print details of calculations. Useful for debugging.
    broadcast : bool
        If True, broadcast conditions against each other. This will compute all combinations.
        If False, each condition should be an equal-length list (or single-valued).
        Disabling broadcasting is useful for calculating equilibrium at selected conditions,
        when those conditions don't comprise a grid.
    calc_opts : dict, optional
        Keyword arguments to pass to `calculate`, the energy/property calculation routine.
    scheduler : Dask scheduler, optional
        Job scheduler for performing the computation.
        If None, return a Dask graph of the computation instead of actually doing it.
    parameters : dict, optional
        Maps SymPy Symbol to numbers, for overriding the values of parameters in the Database.
    solver : pycalphad.core.solver.SolverBase
        Instance of a solver that is used to calculate local equilibria.
        Defaults to a pycalphad.core.solver.InteriorPointSolver.
    callables : dict, optional
        Pre-computed callable functions for equilibrium calculation.

    Returns
    -------
    Structured equilibrium calculation, or Dask graph if scheduler=None.

    Examples
    --------
    None yet.
    """
    if not broadcast:
        raise NotImplementedError('Broadcasting cannot yet be disabled')
    comps = sorted(unpack_components(dbf, comps))
    phases = unpack_phases(phases) or sorted(dbf.phases.keys())
    # remove phases that cannot be active
    list_of_possible_phases = filter_phases(dbf, comps)
    active_phases = sorted(set(list_of_possible_phases).intersection(set(phases)))
    if len(list_of_possible_phases) == 0:
        raise ConditionError('There are no phases in the Database that can be active with components {0}'.format(comps))
    if len(active_phases) == 0:
        raise ConditionError('None of the passed phases ({0}) are active. List of possible phases: {1}.'.format(phases, list_of_possible_phases))
    if isinstance(comps, (str, v.Species)):
        comps = [comps]
    if len(set(comps) - set(dbf.species)) > 0:
        raise EquilibriumError('Components not found in database: {}'
                               .format(','.join([c.name for c in (set(comps) - set(dbf.species))])))
    calc_opts = calc_opts if calc_opts is not None else dict()
    solver = solver if solver is not None else InteriorPointSolver(verbose=verbose)
    parameters = parameters if parameters is not None else dict()
    if isinstance(parameters, dict):
        parameters = OrderedDict(sorted(parameters.items(), key=str))
    models = instantiate_models(dbf, comps, active_phases, model=model, parameters=parameters)
    # Temporary solution until constraint system improves
    if conditions.get(v.N) is None:
        conditions[v.N] = 1
    if np.any(np.array(conditions[v.N]) != 1):
        raise ConditionError('N!=1 is not yet supported, got N={}'.format(conditions[v.N]))
    # Modify conditions values to be within numerical limits, e.g., X(AL)=0
    # Also wrap single-valued conditions with lists
    conds = _adjust_conditions(conditions)

    for cond in conds.keys():
        if isinstance(cond, (v.Composition, v.ChemicalPotential)) and cond.species not in comps:
            raise ConditionError('{} refers to non-existent component'.format(cond))
    state_variables = sorted(get_state_variables(models=models, conds=conds), key=str)
    str_conds = OrderedDict((str(key), value) for key, value in conds.items())
    num_calcs = np.prod([len(i) for i in str_conds.values()])
    components = [x for x in sorted(comps)]
    desired_active_pure_elements = [list(x.constituents.keys()) for x in components]
    desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements for el in constituents]
    pure_elements = sorted(set([x for x in desired_active_pure_elements if x != 'VA']))
    if verbose:
        print('Components:', ' '.join([str(x) for x in comps]))
        print('Phases:', end=' ')
    output = output if output is not None else 'GM'
    output = output if isinstance(output, (list, tuple, set)) else [output]
    output = set(output)
    output |= {'GM'}
    output = sorted(output)
    phase_records = build_phase_records(dbf, comps, active_phases, conds, models,
                                        output='GM', callables=callables,
                                        parameters=parameters, verbose=verbose,
                                        build_gradients=True, build_hessians=True)
    if verbose:
        print('[done]', end='\n')

    # 'calculate' accepts conditions through its keyword arguments
    grid_opts = calc_opts.copy()
    statevar_strings = [str(x) for x in state_variables]
    grid_opts.update({key: value for key, value in str_conds.items() if key in statevar_strings})
    if 'pdens' not in grid_opts:
        grid_opts['pdens'] = 500
    grid = delayed(calculate, pure=False)(dbf, comps, active_phases,
                                          model=models, fake_points=True,
                                          callables=callables, output='GM',
                                          parameters=parameters, **grid_opts)
    coord_dict = str_conds.copy()
    coord_dict['vertex'] = np.arange(
        len(pure_elements) + 1)  # +1 is to accommodate the degenerate degree of freedom at the invariant reactions
    coord_dict['component'] = pure_elements
    grid_shape = tuple(len(x) for x in conds.values()) + (len(pure_elements)+1,)
    properties = delayed(starting_point, pure=False)(conds, state_variables, phase_records, grid)
    conditions_per_chunk_per_axis = 2
    if num_calcs > 1:
        # Generate slices of 'properties'
        slices = []
        for val in grid_shape[:-1]:
            idx_arr = list(range(val))
            num_chunks = int(np.floor(val/conditions_per_chunk_per_axis))
            if num_chunks > 0:
                cond_slices = [x for x in np.array_split(np.asarray(idx_arr), num_chunks) if len(x) > 0]
            else:
                cond_slices = [idx_arr]
            slices.append(cond_slices)
        chunk_dims = [len(slc) for slc in slices]
        chunk_grid = np.array(np.unravel_index(np.arange(np.prod(chunk_dims)), chunk_dims)).T
        res = []
        for chunk in chunk_grid:
            prop_slice = properties[OrderedDict(list(zip(str_conds.keys(),
                                                         [np.atleast_1d(sl)[ch] for ch, sl in zip(chunk, slices)])))]
            job = delayed(_solve_eq_at_conditions, pure=False)(comps, prop_slice, phase_records, grid,
                                                               list(str_conds.keys()), state_variables, verbose, solver=solver)
            res.append(job)
        properties = delayed(_merge_property_slices, pure=False)(properties, chunk_grid, slices, list(str_conds.keys()), res)
    else:
        # Single-process job; don't create child processes
        properties = delayed(_solve_eq_at_conditions, pure=False)(comps, properties, phase_records, grid,
                                                                  list(str_conds.keys()), state_variables, verbose, solver=solver)

    # Compute equilibrium values of any additional user-specified properties
    # We already computed these properties so don't recompute them
    output = sorted(set(output) - {'GM', 'MU'})
    for out in output:
        if (out is None) or (len(out) == 0):
            continue
        # TODO: How do we know if a specified property should be per_phase or not?
        # For now, we make a best guess
        if (out == 'degree_of_ordering') or (out == 'DOO'):
            per_phase = True
        else:
            per_phase = False
        eqcal = delayed(_eqcalculate, pure=False)(dbf, comps, active_phases, conditions, out,
                                                  data=properties, per_phase=per_phase,
                                                  callables=callables,
                                                  parameters=parameters,
                                                  model=models, **calc_opts)
        properties = delayed(properties.merge, pure=False)(eqcal, compat='equals')
    if scheduler is not None:
        properties = dask.compute(properties, scheduler=scheduler)[0]
    properties.attrs['created'] = datetime.utcnow().isoformat()
    if len(kwargs) > 0:
        warnings.warn('The following equilibrium keyword arguments were passed, but unused:\n{}'.format(kwargs))
    return properties
