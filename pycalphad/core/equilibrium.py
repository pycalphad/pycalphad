"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""
from __future__ import print_function
import warnings
import pycalphad.variables as v
from pycalphad.core.utils import unpack_kwarg
from pycalphad.core.utils import unpack_condition, unpack_phases
from pycalphad import calculate, Model
from pycalphad.core.lower_convex_hull import lower_convex_hull
from pycalphad.core.sympydiff_utils import build_functions as compiled_build_functions
from pycalphad.core.phase_rec import PhaseRecord_from_cython, PhaseRecord_from_compiledmodel
from pycalphad.core.compiled_model import CompiledModel
from pycalphad.core.calculate import FallbackModel
from pycalphad.core.constants import MIN_SITE_FRACTION
from pycalphad.core.eqsolver import _solve_eq_at_conditions
from sympy import Add, Symbol
import dask
from dask import delayed
import dask.multiprocessing, dask.async
from xarray import Dataset
import numpy as np
from collections import namedtuple, OrderedDict
from datetime import datetime


class EquilibriumError(Exception):
    "Exception related to calculation of equilibrium."
    pass


class ConditionError(EquilibriumError):
    "Exception related to equilibrium conditions."
    pass


def _adjust_conditions(conds):
    "Adjust conditions values to be within the numerical limit of the solver."
    new_conds = OrderedDict()
    for key, value in sorted(conds.items(), key=str):
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

def _eqcalculate(dbf, comps, phases, conditions, output, data=None, per_phase=False, **kwargs):
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
    indep_vars = ['P', 'T']
    # TODO: Rewrite this to use the coord dict from 'data'
    str_conds = OrderedDict((str(key), value) for key, value in conds.items())
    indep_vals = list([float(x) for x in np.atleast_1d(val)]
                      for key, val in str_conds.items() if key in indep_vars)
    coord_dict = str_conds.copy()
    components = [x for x in sorted(comps) if not x.startswith('VA')]
    coord_dict['vertex'] = np.arange(len(components))
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
        calcres = calculate(dbf, comps, [phase], output=output,
                            points=points, broadcast=False, **statevars)
        result[output].values[np.nonzero(current_phase_indices)] = calcres[output].values
    if not per_phase:
        result[output] = (result[output] * data['NP']).sum(dim='vertex', skipna=True)
    else:
        result['Phase'] = data['Phase'].copy()
        result['NP'] = data['NP'].copy()
    return result

def equilibrium(dbf, comps, phases, conditions, output=None, model=None,
                verbose=False, broadcast=True, calc_opts=None,
                scheduler=dask.async.get_sync,
                parameters=None, **kwargs):
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

    Returns
    -------
    Structured equilibrium calculation, or Dask graph if scheduler=None.

    Examples
    --------
    None yet.
    """
    if not broadcast:
        raise NotImplementedError('Broadcasting cannot yet be disabled')
    from pycalphad import __version__ as pycalphad_version
    active_phases = unpack_phases(phases) or sorted(dbf.phases.keys())
    comps = sorted(comps)
    if len(set(comps) - set(dbf.elements)) > 0:
        raise EquilibriumError('Components not found in database: {}'.format(','.join(set(comps) - set(dbf.elements))))
    indep_vars = ['T', 'P']
    calc_opts = calc_opts if calc_opts is not None else dict()
    model = model if model is not None else FallbackModel
    phase_records = dict()
    diagnostic = kwargs.pop('_diagnostic', False)
    callable_dict = kwargs.pop('callables', dict())
    grad_callable_dict = kwargs.pop('grad_callables', dict())
    hess_callable_dict = kwargs.pop('hess_callables', dict())
    parameters = parameters if parameters is not None else dict()
    if isinstance(parameters, dict):
        parameters = OrderedDict(sorted(parameters.items(), key=str))
    param_symbols = tuple(parameters.keys())
    param_values = np.atleast_1d(np.array(list(parameters.values()), dtype=np.float))
    maximum_internal_dof = 0
    # Modify conditions values to be within numerical limits, e.g., X(AL)=0
    # Also wrap single-valued conditions with lists
    conds = _adjust_conditions(conditions)
    for cond in conds.keys():
        if isinstance(cond, (v.Composition, v.ChemicalPotential)) and cond.species not in comps:
            raise ConditionError('{} refers to non-existent component'.format(cond))
    str_conds = OrderedDict((str(key), value) for key, value in conds.items())
    num_calcs = np.prod([len(i) for i in str_conds.values()])
    build_functions = compiled_build_functions
    backend_mode = 'compiled'
    if kwargs.get('_backend', None):
        backend_mode = kwargs['_backend']
    if verbose:
        backend_dict = {'compiled': 'Compiled (autowrap)', 'interpreted': 'Interpreted (autograd)'}
        print('Calculation Backend: {}'.format(backend_dict.get(backend_mode, 'Custom')))
    indep_vals = list([float(x) for x in np.atleast_1d(val)]
                      for key, val in str_conds.items() if key in indep_vars)
    components = [x for x in sorted(comps) if not x.startswith('VA')]
    # Construct models for each phase; prioritize user models
    models = unpack_kwarg(model, default_arg=FallbackModel)
    if verbose:
        print('Components:', ' '.join(comps))
        print('Phases:', end=' ')
    max_phase_name_len = max(len(name) for name in active_phases)
    # Need to allow for '_FAKE_' psuedo-phase
    max_phase_name_len = max(max_phase_name_len, 6)
    for name in active_phases:
        mod = models[name]
        if isinstance(mod, type):
            models[name] = mod = mod(dbf, comps, name, parameters=parameters)
        if isinstance(mod, CompiledModel):
            phase_records[name.upper()] = PhaseRecord_from_compiledmodel(mod, param_values)
            maximum_internal_dof = max(maximum_internal_dof, sum(mod.sublattice_dof))
        else:
            site_fracs = mod.site_fractions
            variables = sorted(site_fracs, key=str)
            maximum_internal_dof = max(maximum_internal_dof, len(site_fracs))
            out = models[name].energy
            if (not callable_dict.get(name, False)) or not (grad_callable_dict.get(name, False)) \
                    or (not hess_callable_dict.get(name, False)):
                # Only force undefineds to zero if we're not overriding them
                undefs = list(out.atoms(Symbol) - out.atoms(v.StateVariable) - set(param_symbols))
                for undef in undefs:
                    out = out.xreplace({undef: float(0)})
                cf, gf, hf = build_functions(out, tuple([v.P, v.T] + site_fracs), parameters=param_symbols)
                if callable_dict.get(name, None) is None:
                    callable_dict[name] = cf
                if grad_callable_dict.get(name, None) is None:
                    grad_callable_dict[name] = gf
                if hess_callable_dict.get(name, None) is None:
                    hess_callable_dict[name] = hf

            phase_records[name.upper()] = PhaseRecord_from_cython(comps, variables,
                                                                np.array(dbf.phases[name].sublattices, dtype=np.float),
                                                                param_values, callable_dict[name],
                                                                grad_callable_dict[name], hess_callable_dict[name])
        if verbose:
            print(name, end=' ')
    if verbose:
        print('[done]', end='\n')

    # 'calculate' accepts conditions through its keyword arguments
    grid_opts = calc_opts.copy()
    grid_opts.update({key: value for key, value in str_conds.items() if key in indep_vars})
    if 'pdens' not in grid_opts:
        grid_opts['pdens'] = 500

    coord_dict = str_conds.copy()
    coord_dict['vertex'] = np.arange(len(components))
    grid_shape = np.meshgrid(*coord_dict.values(),
                             indexing='ij', sparse=False)[0].shape
    coord_dict['component'] = components

    grid = delayed(calculate, pure=False)(dbf, comps, active_phases, output='GM',
                                          model=models, callables=callable_dict, fake_points=True,
                                          parameters=parameters, **grid_opts)

    properties = delayed(Dataset, pure=False)({'NP': (list(str_conds.keys()) + ['vertex'],
                                                      np.empty(grid_shape)),
                                               'GM': (list(str_conds.keys()),
                                                      np.empty(grid_shape[:-1])),
                                               'MU': (list(str_conds.keys()) + ['component'],
                                                      np.empty(grid_shape)),
                                               'X': (list(str_conds.keys()) + ['vertex', 'component'],
                                                     np.empty(grid_shape + (grid_shape[-1],))),
                                               'Y': (list(str_conds.keys()) + ['vertex', 'internal_dof'],
                                                     np.empty(grid_shape + (maximum_internal_dof,))),
                                               'Phase': (list(str_conds.keys()) + ['vertex'],
                                                         np.empty(grid_shape, dtype='U%s' % max_phase_name_len)),
                                               'points': (list(str_conds.keys()) + ['vertex'],
                                                          np.empty(grid_shape, dtype=np.int32))
                                               },
                                              coords=coord_dict,
                                              attrs={'engine': 'pycalphad %s' % pycalphad_version},
                                              )
    # One last call to ensure 'properties' and 'grid' are consistent with one another
    properties = delayed(lower_convex_hull, pure=False)(grid, properties)
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
                                                              list(str_conds.keys()), verbose)
            res.append(job)
        properties = delayed(_merge_property_slices, pure=False)(properties, chunk_grid, slices, list(str_conds.keys()), res)
    else:
        # Single-process job; don't create child processes
        properties = delayed(_solve_eq_at_conditions, pure=False)(comps, properties, phase_records, grid,
                                                                 list(str_conds.keys()), verbose)

    # Compute equilibrium values of any additional user-specified properties
    output = output if isinstance(output, (list, tuple, set)) else [output]
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
        for phase_name, mod in models.items():
            if isinstance(mod, CompiledModel) or isinstance(mod, FallbackModel):
                models[phase_name] = Model(dbf, comps, phase_name, parameters=parameters)
        eqcal = delayed(_eqcalculate, pure=False)(dbf, comps, active_phases, conditions, out,
                                                  data=properties, per_phase=per_phase, model=models, **calc_opts)
        properties = delayed(properties.merge, pure=False)(eqcal, inplace=True, compat='equals')
    if scheduler is not None:
        properties = dask.compute(properties, get=scheduler)[0]
    properties.attrs['created'] = datetime.utcnow().isoformat()
    if len(kwargs) > 0:
        warnings.warn('The following equilibrium keyword arguments were passed, but unused:\n{}'.format(kwargs))
    return properties
