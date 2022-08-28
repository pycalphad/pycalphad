from typing_extensions import runtime
import warnings
from collections import OrderedDict, Counter
from collections.abc import Mapping
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
import numpy.typing as npt
from typing import cast, Dict, Union, List, Optional, Tuple, Protocol, runtime_checkable

from pycalphad.io.database import Database
from pycalphad.variables import Species, StateVariable

from runtype import dataclass, isa
from dataclasses import field
from typing import Sequence, Mapping, Tuple, Union
from copy import copy


def _adjust_conditions(conds) -> 'OrderedDict[StateVariable, List[float]]':
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

@runtime_checkable
class ComputableProperty(Protocol):
    def compute_property(self, compsets: List[CompositionSet], cur_conds: Dict[str, float], chemical_potentials: npt.ArrayLike) -> npt.ArrayLike:
        ...
    @property
    def shape(self) -> Tuple[int]:
        ...

class ComputedProperty(object):
    def __init__(self, model_attr_name: str, phase_name: Optional[str] = None):
        self.model_attr_name = model_attr_name
        self.phase_name = phase_name

    def expand_wildcard(self, phase_names):
        return [self.__class__(self.model_attr_name, phase_name) for phase_name in phase_names]

    def __str__(self):
        result = self.model_attr_name
        if self.phase_name is not None:
            result += f'({self.phase_name})'
        return result

    @property
    def shape(self):
        # Need to distinguish between HM, HM(*)
        return (1,)

    def dims(self, compsets: List[CompositionSet]):
        return ('phase',)

    def compute_property(self, compsets: List[CompositionSet], cur_conds: Dict[str, float], chemical_potentials: npt.ArrayLike) -> npt.ArrayLike:
        if self.phase_name is None:
            return np.nansum([compset.NP*self.compute_per_phase_property(compset, cur_conds) for compset in compsets])
        else:
            tokens = self.phase_name.split('#')
            phase_name = tokens[0]
            if len(tokens) > 1:
                multiplicity = int(tokens[1])
            else:
                multiplicity = 1
            multiplicity_seen = 0
            for compset in compsets:
                if compset.phase_record.phase_name != phase_name:
                    continue
                multiplicity_seen += 1
                if multiplicity == multiplicity_seen:
                    return self.compute_per_phase_property(compset, cur_conds)
            return np.atleast_1d(np.nan)

    def compute_per_phase_property(self, compset: CompositionSet, cur_conds: Dict[str, float]) -> npt.ArrayLike:
        out = np.atleast_1d(np.zeros(1))
        compset.phase_record.prop(out, compset.dof, self.model_attr_name.encode('utf-8'))
        return out

def make_computable_property(inp: Union[str, ComputableProperty]) -> ComputableProperty:
    if isa(inp, ComputableProperty):
        return inp
    try:
        begin_parens = inp.index('(')
        end_parens = inp.index(')')
    except ValueError:
        begin_parens = len(inp)
        end_parens = len(inp)

    specified_prop = inp[:begin_parens].strip()

    # TODO: Add support for '.' in dot derivative
    prop = getattr(v, specified_prop, None)
    if prop is None:
        prop = COMPUTED_PROPERTIES.get(specified_prop, ComputedProperty)
    if begin_parens != end_parens:
        specified_args = tuple(x.strip() for x in inp[begin_parens+1:end_parens].split(','))
        if not isinstance(prop, type):
            prop_instance = type(prop)(*((specified_prop,)+specified_args))
            print('first branch')
        else:
            if issubclass(prop, StateVariable):
                prop_instance = prop(*(specified_args))
            else:
                prop_instance = prop(*((specified_prop,)+specified_args))
            print('second branch')
    else:
        print('third branch')
        if isinstance(prop, type):
            prop = prop(specified_prop)
            print('three.1')
        prop_instance = prop
    print(f'returning {prop_instance}')
    return prop_instance

class DotDerivativeComputedProperty:
    def __init__(self, model_attr_name):
        pass

    @property
    def shape(self):
        # Need to distinguish between HM, HM(*)
        return (1,)
    
    def compute_property(self, compsets, cur_conds, chemical_potentials):
        solver = Solver()
        spec = solver.get_system_spec(compsets, cur_conds)
        state = spec.get_new_state(compsets)
        state.chemical_potentials[:] = chemical_potentials
        state.recompute(spec)
        return dot_derivative(spec, state, 'HM', v.T)

COMPUTED_PROPERTIES = {}
COMPUTED_PROPERTIES['DOO'] = ComputedProperty('DOO')
COMPUTED_PROPERTIES['degree_of_ordering'] = COMPUTED_PROPERTIES['DOO']
COMPUTED_PROPERTIES['heat_capacity'] = DotDerivativeComputedProperty('heat_capacity')

class PhaseName:
    @classmethod
    def cast_from(cls, s: str) -> "PhaseName":
        return s.upper()

class ConditionValue:
    @classmethod
    def cast_from(cls, value: Union[float, Sequence[float]]) -> "ConditionValue":
        return unpack_condition(value)


@dataclass(check_types='cast', frozen=False)
class Workspace:
    dbf: Database
    comps: Sequence[Species]
    phases: Sequence[PhaseName]
    conditions: Mapping[StateVariable, ConditionValue]
    verbose: Optional[bool] = False
    models: Optional[Union[Model, Mapping[PhaseName, Model]]] = None
    phase_record_factory: Optional[PhaseRecordFactory] = None
    parameters: Optional[OrderedDict] = field(default_factory=dict)
    calc_opts: Optional[Mapping] = field(default_factory=dict)
    solver: Optional[Solver] = None
    ndim: int = field(init=False, default=0)
    eq: Optional[LightDataset] = field(init=False, default=None)

    def __post_init__(self):
        self.comps = sorted(unpack_components(self.dbf, self.comps))
        self.phases = unpack_phases(self.phases) or sorted(self.dbf.phases.keys())
        list_of_possible_phases = filter_phases(self.dbf, self.comps)
        if len(list_of_possible_phases) == 0:
            raise ConditionError('There are no phases in the Database that can be active with components {0}'.format(self.comps))
        self.active_phases = filter_phases(self.dbf, self.comps, self.phases)
        if len(self.active_phases) == 0:
            raise ConditionError('None of the passed phases ({0}) are active. List of possible phases: {1}.'.format(self.phases, list_of_possible_phases))
        if len(set(self.comps) - set(self.dbf.species)) > 0:
            raise EquilibriumError('Components not found in database: {}'
                                .format(','.join([c.name for c in (set(self.comps) - set(self.dbf.species))])))
        self.solver = self.solver if self.solver is not None else Solver(verbose=self.verbose)
        if isinstance(self.parameters, dict):
            self.parameters = OrderedDict(sorted(self.parameters.items(), key=str))
        # Temporary solution until constraint system improves
        if self.conditions.get(v.N) is None:
            self.conditions[v.N] = 1
        if np.any(np.array(self.conditions[v.N]) != 1):
            raise ConditionError('N!=1 is not yet supported, got N={}'.format(self.conditions[v.N]))
        # Modify conditions values to be within numerical limits, e.g., X(AL)=0
        # Also wrap single-valued conditions with lists
        conds = _adjust_conditions(self.conditions)

        for cond in conds.keys():
            if isinstance(cond, (v.MoleFraction, v.ChemicalPotential)) and cond.species not in self.comps:
                raise ConditionError('{} refers to non-existent component'.format(cond))
        self.ndim = 0
        for cond_val in conds.values():
            if len(cond_val) > 1:
                self.ndim += 1
        str_conds = OrderedDict((str(key), value) for key, value in conds.items())
        components = [x for x in sorted(self.comps)]
        desired_active_pure_elements = [list(x.constituents.keys()) for x in components]
        desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements for el in constituents]
        pure_elements = sorted(set([x for x in desired_active_pure_elements if x != 'VA']))

        if self.phase_record_factory is None:
            self.models = instantiate_models(self.dbf, self.comps, self.active_phases, model=self.models, parameters=self.parameters)
            self.phase_record_factory = PhaseRecordFactory(self.dbf, self.comps, conds, self.models, parameters=self.parameters)
        else:
            # phase_records were provided, instantiated models must also be provided by the caller
            if not isinstance(self.models, Mapping):
                raise ValueError("A dictionary of instantiated models must be passed to `equilibrium` with the `model` argument if the `phase_records` argument is used.")
            active_phases_without_models = [name for name in self.active_phases if not isinstance(self.models.get(name), Model)]
            if len(active_phases_without_models) > 0:
                raise ValueError(f"model must contain a Model instance for every active phase. Missing Model objects for {sorted(active_phases_without_models)}")

        self.phase_record_factory.param_values[:] = list(self.parameters.values())

        state_variables = list(self.phase_record_factory.values())[0].state_variables

        # 'calculate' accepts conditions through its keyword arguments
        grid_opts = self.calc_opts.copy()
        statevar_strings = [str(x) for x in state_variables]
        grid_opts.update({key: value for key, value in str_conds.items() if key in statevar_strings})

        if 'pdens' not in grid_opts:
            grid_opts['pdens'] = 60
        grid = calculate(self.dbf, self.comps, self.active_phases, model=self.models, fake_points=True,
                        phase_records=self.phase_record_factory, output='GM', parameters=self.parameters,
                        to_xarray=False, **grid_opts)
        coord_dict = str_conds.copy()
        coord_dict['vertex'] = np.arange(len(pure_elements) + 1)  # +1 is to accommodate the degenerate degree of freedom at the invariant reactions
        coord_dict['component'] = pure_elements
        properties = starting_point(conds, state_variables, self.phase_record_factory, grid)
        self.eq = _solve_eq_at_conditions(properties, self.phase_record_factory, grid,
                                          list(str_conds.keys()), state_variables,
                                          self.verbose, solver=self.solver)

    def _detect_phase_multiplicity(self):
        multiplicity = {k: 0 for k in sorted(self.phase_record_factory.keys())}
        prop_GM_values = self.eq.GM
        prop_Phase_values = self.eq.Phase
        for index in np.ndindex(prop_GM_values.shape):
            cur_multiplicity = Counter()
            for phase_name in prop_Phase_values[index]:
                if phase_name == '' or phase_name == '_FAKE_':
                    continue
                cur_multiplicity[phase_name] += 1
            for key, value in cur_multiplicity.items():
                multiplicity[key] = max(multiplicity[key], value)
        return multiplicity

    def get(self, *args: Tuple[ComputableProperty], values_only=True):
        if self.ndim > 1:
            raise ValueError('Dimension of calculation is greater than one')
        multiplicity = self._detect_phase_multiplicity()
        args = list(map(make_computable_property, args))
        
        indices_to_delete = []
        i = 0
        while i < len(args):
            if hasattr(args[i], 'phase_name') and args[i].phase_name == '*':
                print('hit phase name branch')
                indices_to_delete.append(i)
                phase_names = sorted(self.phase_record_factory.keys())
                additional_args = args[i].expand_wildcard(phase_names=phase_names)
                args.extend(additional_args)
            elif hasattr(args[i], 'species') and args[i].species == v.Species('*'):
                print(f'hit species branch with {args[i]=}')
                indices_to_delete.append(i)
                internal_to_phase = hasattr(args[i], 'sublattice_index')
                if internal_to_phase:
                    components = [x for x in self.phase_record_factory[args[i].phase_name].variables
                                  if x.sublattice_index == args[i].sublattice_index]
                else:
                    components = self.phase_record_factory[args[i].phase_name].nonvacant_elements
                additional_args = args[i].expand_wildcard(components=components)
                args.extend(additional_args)
            else:
                # This is a concrete ComputableProperty
                if hasattr(args[i], 'phase_name') and (args[i].phase_name is not None) \
                    and not ('#' in args[i].phase_name) and multiplicity[args[i].phase_name] > 1:
                    # Miscibility gap detected; expand property into multiple composition sets
                    additional_phase_names = [args[i].phase_name+'#'+str(multi_idx+1)
                                              for multi_idx in range(multiplicity[args[i].phase_name])]
                    indices_to_delete.append(i)
                    additional_args = args[i].expand_wildcard(phase_names=additional_phase_names)
                    args.extend(additional_args)
            i += 1
        
        # Watch deletion order! Indices will change as items are deleted
        for deletion_index in reversed(indices_to_delete):
            del args[deletion_index]
        print(f'{args=}')
        arr_size = self.eq.GM.size
        results = dict()

        prop_GM_values = self.eq.GM
        prop_Y_values = self.eq.Y
        prop_MU_values = self.eq.MU
        prop_NP_values = self.eq.NP
        prop_Phase_values = self.eq.Phase
        conds_keys = [str(k) for k in self.eq.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
        state_variables = list(self.phase_record_factory.values())[0].state_variables
        str_state_variables = [str(k) for k in state_variables]
        local_index = 0

        for index in np.ndindex(prop_GM_values.shape):
            cur_conds = OrderedDict(zip(conds_keys,
                                        [np.asarray(self.eq.coords[b][a], dtype=np.float_)
                                        for a, b in zip(index, conds_keys)]))
            state_variable_values = [cur_conds[key] for key in str_state_variables]
            state_variable_values = np.array(state_variable_values)
            composition_sets = []
            for phase_idx, phase_name in enumerate(prop_Phase_values[index]):
                if phase_name == '' or phase_name == '_FAKE_':
                    continue
                phase_record = self.phase_record_factory[phase_name]
                sfx = prop_Y_values[index + np.index_exp[phase_idx, :phase_record.phase_dof]]
                phase_amt = prop_NP_values[index + np.index_exp[phase_idx]]
                compset = CompositionSet(phase_record)
                compset.update(sfx, phase_amt, state_variable_values)
                composition_sets.append(compset)
            chemical_potentials = prop_MU_values[index]
            
            for arg in args:
                if results.get(arg, None) is None:
                    results[arg] = np.zeros((arr_size,) + arg.shape)
                results[arg][local_index, :] = arg.compute_property(composition_sets, cur_conds, chemical_potentials)
            local_index += 1
        
        if values_only:
            return list(results.values())
        else:
            return results
    
    def plot(self, x: ComputableProperty, *ys: Tuple[ComputableProperty], ax=None):
        import matplotlib.pyplot as plt
        ax = ax if ax is not None else plt.gca()
        x = make_computable_property(x)
        data = self.get(x, *ys, values_only=False)
        
        for y in data.keys():
            if y == x:
                continue
            ax.plot(data[x], data[y], label=str(y))
        ax.legend()

# Upstream bug: Types are not cast before setattr is called, when frozen=False
def _setattr_workaround(self, name, value):
    try:
        field = self.__dataclass_fields__[name]
    except (KeyError, AttributeError):
        pass
    else:
        if not isa(value, field.type):
            value = field.type.cast_from(value)
    object.__setattr__(self, name, value)
Workspace.__setattr__ = _setattr_workaround