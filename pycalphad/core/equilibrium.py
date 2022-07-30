"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from datetime import datetime
import pycalphad.variables as v
from pycalphad.core.utils import unpack_components, unpack_condition, unpack_phases, filter_phases, instantiate_models, get_state_variables
from pycalphad import calculate
from pycalphad.core.errors import EquilibriumError, ConditionError
from pycalphad.core.starting_point import starting_point
from pycalphad.codegen.callables import PhaseRecordFactory
from pycalphad.core.eqsolver import _solve_eq_at_conditions
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.solver import Solver
from pycalphad.core.minimizer import site_fraction_differential, state_variable_differential
from pycalphad.core.light_dataset import LightDataset
from pycalphad.model import Model
import numpy as np


def _adjust_conditions(conds):
    "Adjust conditions values to be within the numerical limit of the solver."
    new_conds = OrderedDict()
    minimum_composition = 1e-10
    for key, value in sorted(conds.items(), key=str):
        if key == str(key):
            key = getattr(v, key, key)
        if isinstance(key, v.MoleFraction):
            vals = unpack_condition(value)
            # "Zero" composition is a common pattern. Do not warn for that case.
            if np.any(np.logical_and(np.asarray(vals) < minimum_composition, np.asarray(vals) > 0)):
                warnings.warn(
                    f"Some specified compositions are below the minimum allowed composition of {minimum_composition}.")
            new_conds[key] = [max(val, minimum_composition) for val in vals]
        else:
            new_conds[key] = unpack_condition(value)
    return new_conds


def dot_derivative(spec, state, property_of_interest, statevar_of_interest):
    """
    Sample the internal degrees of freedom of a phase.

    Parameters
    ----------
    spec : SystemSpecification
        some description
    state : SystemState
        another description
    property_of_interest : string
    statevar_of_interest : StateVariable
    Returns
    -------
    dot derivative of property
    """
    property_of_interest = str(property_of_interest).encode('utf-8')
    state_variables = state.compsets[0].phase_record.state_variables
    statevar_idx = sorted(state_variables, key=str).index(statevar_of_interest)
    delta_chemical_potentials, delta_statevars, delta_phase_amounts = \
    state_variable_differential(spec, state, statevar_idx)

    # Sundman et al, 2015, Eq. 73
    dot_derivative = 0.0
    naive_derivative = 0.0
    for idx, compset in enumerate(state.compsets):
        func_value = np.atleast_1d(np.zeros(1))
        grad_value = np.zeros(compset.dof.shape[0])
        compset.phase_record.prop(func_value, compset.dof, property_of_interest)
        compset.phase_record.prop_grad(grad_value, compset.dof, property_of_interest)
        delta_sitefracs = site_fraction_differential(state.cs_states[idx], delta_chemical_potentials,
                                                     delta_statevars)

        dot_derivative += delta_phase_amounts[idx] * func_value[0]
        dot_derivative += compset.NP * grad_value[statevar_idx] * delta_statevars[statevar_idx]
        naive_derivative += compset.NP * grad_value[statevar_idx] * delta_statevars[statevar_idx]
        dot_derivative += compset.NP * np.dot(delta_sitefracs, grad_value[len(state_variables):])

    return dot_derivative


def apply_to_dataset(input_dataset, phase_records, function_to_apply, per_phase=False, fill_value=np.nan, dtype=float):
    prop_NP_values = input_dataset.NP
    prop_Phase_values = input_dataset.Phase
    prop_Y_values = input_dataset.Y
    prop_MU_values = input_dataset.MU
    prop_GM_values = input_dataset.GM
    conds_keys = [str(k) for k in input_dataset.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
    state_variables = list(phase_records.values())[0].state_variables
    str_state_variables = [str(k) for k in state_variables]
    if per_phase:
        output_array = np.full(prop_NP_values.shape, fill_value, dtype=dtype)
    else:
        output_array = np.full(prop_GM_values.shape, fill_value, dtype=dtype)

    for index in np.ndindex(prop_GM_values.shape):
        cur_conds = OrderedDict(zip(conds_keys,
                                    [np.asarray(input_dataset.coords[b][a], dtype=np.float_)
                                     for a, b in zip(index, conds_keys)]))
        state_variable_values = [cur_conds[key] for key in str_state_variables]
        state_variable_values = np.array(state_variable_values)
        composition_sets = []
        for phase_idx, phase_name in enumerate(prop_Phase_values[index]):
            if phase_name == '' or phase_name == '_FAKE_':
                continue
            phase_record = phase_records[phase_name]
            sfx = prop_Y_values[index + np.index_exp[phase_idx, :phase_record.phase_dof]]
            phase_amt = prop_NP_values[index + np.index_exp[phase_idx]]
            compset = CompositionSet(phase_record)
            compset.update(sfx, phase_amt, state_variable_values)
            if per_phase:
                augmented_index = index + np.index_exp[phase_idx]
                output_array[augmented_index] = function_to_apply(compset, cur_conds, augmented_index)
            else:
                composition_sets.append(compset)
        if not per_phase:
            chemical_potentials = prop_MU_values[index]
            output_array[index] = function_to_apply(composition_sets, cur_conds, chemical_potentials, index)
    return output_array


def equilibrium(dbf, comps, phases, conditions, output=None, model=None,
                verbose=False, broadcast=True, calc_opts=None, to_xarray=True,
                parameters=None, solver=None, callables=None,
                phase_records=None, phase_record_factory=None, **kwargs):
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
    to_xarray : bool
        Whether to return an xarray Dataset (True, default) or an EquilibriumResult.
    parameters : dict, optional
        Maps SymEngine Symbol to numbers, for overriding the values of parameters in the Database.
    solver : pycalphad.core.solver.SolverBase
        Instance of a solver that is used to calculate local equilibria.
        Defaults to a pycalphad.core.solver.Solver.
    callables : dict, optional
        Pre-computed callable functions for equilibrium calculation.
    phase_records : Optional[Mapping[str, PhaseRecord]]
        Mapping of phase names to PhaseRecord objects with `'GM'` output. Must include
        all active phases. The `model` argument must be a mapping of phase names to
        instances of Model objects.

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
    list_of_possible_phases = filter_phases(dbf, comps)
    if len(list_of_possible_phases) == 0:
        raise ConditionError('There are no phases in the Database that can be active with components {0}'.format(comps))
    active_phases = filter_phases(dbf, comps, phases)
    if len(active_phases) == 0:
        raise ConditionError('None of the passed phases ({0}) are active. List of possible phases: {1}.'.format(phases, list_of_possible_phases))
    if isinstance(comps, (str, v.Species)):
        comps = [comps]
    if len(set(comps) - set(dbf.species)) > 0:
        raise EquilibriumError('Components not found in database: {}'
                               .format(','.join([c.name for c in (set(comps) - set(dbf.species))])))
    calc_opts = calc_opts if calc_opts is not None else dict()
    solver = solver if solver is not None else Solver(verbose=verbose)
    parameters = parameters if parameters is not None else dict()
    if isinstance(parameters, dict):
        parameters = OrderedDict(sorted(parameters.items(), key=str))
    # Temporary solution until constraint system improves
    if conditions.get(v.N) is None:
        conditions[v.N] = 1
    if np.any(np.array(conditions[v.N]) != 1):
        raise ConditionError('N!=1 is not yet supported, got N={}'.format(conditions[v.N]))
    # Modify conditions values to be within numerical limits, e.g., X(AL)=0
    # Also wrap single-valued conditions with lists
    conds = _adjust_conditions(conditions)

    for cond in conds.keys():
        if isinstance(cond, (v.MoleFraction, v.ChemicalPotential)) and cond.species not in comps:
            raise ConditionError('{} refers to non-existent component'.format(cond))
    str_conds = OrderedDict((str(key), value) for key, value in conds.items())
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
    if phase_record_factory is None:
        models = instantiate_models(dbf, comps, active_phases, model=model, parameters=parameters)
        phase_record_factory = PhaseRecordFactory(dbf, comps, conds, models, parameters=parameters)
    else:
        # phase_records were provided, instantiated models must also be provided by the caller
        models = model
        if not isinstance(models, Mapping):
            raise ValueError("A dictionary of instantiated models must be passed to `equilibrium` with the `model` argument if the `phase_records` argument is used.")
        active_phases_without_models = [name for name in active_phases if not isinstance(models.get(name), Model)]
        if len(active_phases_without_models) > 0:
            raise ValueError(f"model must contain a Model instance for every active phase. Missing Model objects for {sorted(active_phases_without_models)}")

    phase_record_factory.param_values[:] = list(parameters.values())

    if verbose:
        print('[done]', end='\n')

    state_variables = sorted(get_state_variables(models=models, conds=conds), key=str)

    # 'calculate' accepts conditions through its keyword arguments
    grid_opts = calc_opts.copy()
    statevar_strings = [str(x) for x in state_variables]
    grid_opts.update({key: value for key, value in str_conds.items() if key in statevar_strings})

    if 'pdens' not in grid_opts:
        grid_opts['pdens'] = 60
    grid = calculate(dbf, comps, active_phases, model=models, fake_points=True,
                     phase_records=phase_record_factory, output='GM', parameters=parameters,
                     to_xarray=False, **grid_opts)
    coord_dict = str_conds.copy()
    coord_dict['vertex'] = np.arange(len(pure_elements) + 1)  # +1 is to accommodate the degenerate degree of freedom at the invariant reactions
    coord_dict['component'] = pure_elements
    properties = starting_point(conds, state_variables, phase_record_factory, grid)
    properties = _solve_eq_at_conditions(properties, phase_record_factory, grid,
                                         list(str_conds.keys()), state_variables,
                                         verbose, solver=solver)

    # Compute equilibrium values of any additional user-specified properties
    # We already computed these properties so don't recompute them
    output = sorted(set(output) - {'GM', 'MU'})
    for out in output:
        if (out is None) or (len(out) == 0):
            continue
        if isinstance(out, str):
            cprop = COMPUTED_PROPERTIES[out]
            if isinstance(cprop, type):
                cprop = cprop(out)
        else:
            cprop = out
            out = str(cprop)
        result_array = cprop.compute(properties, phase_record_factory)
        prop_dims = list(str_conds.keys())
        if cprop.per_phase:
            prop_dims.append('vertex')
        result = LightDataset({out: (prop_dims, result_array)}, coords=coord_dict)
        properties.merge(result, inplace=True, compat='equals')
    if to_xarray:
        properties = properties.get_dataset()
    properties.attrs['created'] = datetime.utcnow().isoformat()
    if len(kwargs) > 0:
        warnings.warn('The following equilibrium keyword arguments were passed, but unused:\n{}'.format(kwargs))
    return properties

class ComputedProperty(object):
    def __init__(self, model_attr_name, per_phase=False):
        self.per_phase = per_phase
        self.model_attr_name = model_attr_name

        if self.per_phase:
            self._apply_func = self.calculate_per_phase_property
        else:
            self._apply_func = self.calculate_system_property

    def __str__(self):
        return self.model_attr_name

    def calculate_system_property(self, compsets, cur_conds, chemical_potentials, index):
        return np.nansum([compset.NP*self.calculate_per_phase_property(compset, cur_conds, index) for compset in compsets])

    def calculate_per_phase_property(self, compset, cur_conds, index):
        out = np.atleast_1d(np.zeros(1))
        compset.phase_record.prop(out, compset.dof, self.model_attr_name.encode('utf-8'))
        return out

    def compute(self, input_dataset, phase_records):
        return apply_to_dataset(input_dataset, phase_records, self._apply_func,
                                per_phase=self.per_phase, fill_value=np.nan, dtype=float)

class DotDerivativeComputedProperty(ComputedProperty):
    def __init__(self, model_attr_name):
        super().__init__(model_attr_name, per_phase=False)
    
    def calculate_system_property(self, compsets, cur_conds, chemical_potentials, index):
        solver = Solver()
        spec = solver.get_system_spec(compsets, cur_conds)
        state = spec.get_new_state(compsets)
        state.chemical_potentials[:] = chemical_potentials
        state.recompute(spec)
        return dot_derivative(spec, state, 'H', v.T)

COMPUTED_PROPERTIES = defaultdict(lambda: ComputedProperty)
COMPUTED_PROPERTIES['DOO'] = ComputedProperty('DOO', per_phase=True)
COMPUTED_PROPERTIES['degree_of_ordering'] = COMPUTED_PROPERTIES['DOO']
COMPUTED_PROPERTIES['heat_capacity'] = DotDerivativeComputedProperty('heat_capacity')