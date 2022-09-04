from ast import Str
import warnings
from collections import OrderedDict, Counter
from collections.abc import Mapping
from pycalphad.property_framework.computed_property import DotDerivativeComputedProperty
import pycalphad.variables as v
from pycalphad.core.utils import unpack_components, unpack_condition, unpack_phases, filter_phases, instantiate_models, get_state_variables
from pycalphad import calculate
from pycalphad.core.errors import EquilibriumError, ConditionError
from pycalphad.core.starting_point import starting_point
from pycalphad.codegen.callables import PhaseRecordFactory
from pycalphad.core.eqsolver import _solve_eq_at_conditions
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.solver import Solver, SolverBase
from pycalphad.core.light_dataset import LightDataset
from pycalphad.model import Model
import numpy as np
from typing import Dict, Union, List, Optional, Tuple, Type, Sequence, Mapping
from pycalphad.io.database import Database
from pycalphad.variables import Species, StateVariable
from pycalphad.property_framework import ComputableProperty, as_property
from runtype import dataclass, isa
from dataclasses import field


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

class PhaseList:
    @classmethod
    def cast_from(cls, s: Union[str, Sequence]) -> "PhaseList":
        if isinstance(s, str):
            s = [s]
        return sorted(PhaseName.cast_from(x) for x in s)

class PhaseName:
    @classmethod
    def cast_from(cls, s: str) -> "PhaseName":
        return s.upper()

class ConditionValue:
    @classmethod
    def cast_from(cls, value: Union[float, Sequence[float]]) -> "ConditionValue":
        return unpack_condition(value)

class ConditionKey:
    @classmethod
    def cast_from(cls, key: Union[str, StateVariable]) -> "ConditionKey":
        return as_property(key)


@dataclass(check_types='cast', frozen=False)
class Workspace:
    dbf: Database
    comps: Sequence[Species]
    phases: PhaseList
    conditions: Mapping[ConditionKey, ConditionValue]
    verbose: Optional[bool] = False
    models: Optional[Union[Model, Type[Model], Mapping[PhaseName, Model]]] = None
    phase_record_factory: Optional[PhaseRecordFactory] = None
    parameters: Optional[Dict] = None
    calc_opts: Optional[Dict] = None
    solver: Optional[SolverBase] = None
    ndim: int = 0
    eq: Optional[LightDataset] = None

    def __post_init__(self):
        # XXX: Why isn't default_factory working here?
        if self.parameters is None:
            self.parameters = dict()
        if self.calc_opts is None:
            self.calc_opts = dict()
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

    def _expand_property_arguments(self, args: Sequence[ComputableProperty]):
        "Mutates args"
        multiplicity = self._detect_phase_multiplicity()
        indices_to_delete = []
        i = 0
        while i < len(args):
            if hasattr(args[i], 'phase_name') and args[i].phase_name == '*':
                indices_to_delete.append(i)
                phase_names = sorted(self.phase_record_factory.keys())
                additional_args = args[i].expand_wildcard(phase_names=phase_names)
                args.extend(additional_args)
            elif hasattr(args[i], 'species') and args[i].species == v.Species('*'):
                indices_to_delete.append(i)
                internal_to_phase = hasattr(args[i], 'sublattice_index')
                if internal_to_phase:
                    components = [x for x in self.phase_record_factory[args[i].phase_name].variables
                                  if x.sublattice_index == args[i].sublattice_index]
                else:
                    components = self.phase_record_factory[args[i].phase_name].nonvacant_elements
                additional_args = args[i].expand_wildcard(components=components)
                args.extend(additional_args)
            elif isinstance(args[i], DotDerivativeComputedProperty):
                numerator_args = [args[i].numerator]
                self._expand_property_arguments(numerator_args)
                denominator_args = [args[i].denominator]
                self._expand_property_arguments(denominator_args)
                if (len(numerator_args) > 1) or (len(denominator_args) > 1):
                    for n_arg in numerator_args:
                        for d_arg in denominator_args:
                            args.append(DotDerivativeComputedProperty(n_arg, d_arg))
                    indices_to_delete.append(i)
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

    def enumerate_composition_sets(self):
        if self.eq is None:
            return
        prop_GM_values = self.eq.GM
        prop_Y_values = self.eq.Y
        prop_NP_values = self.eq.NP
        prop_Phase_values = self.eq.Phase
        conds_keys = [str(k) for k in self.eq.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
        state_variables = list(self.phase_record_factory.values())[0].state_variables
        str_state_variables = [str(k) for k in state_variables]

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
            yield index, composition_sets


    def get(self, *args: Tuple[ComputableProperty], values_only=True):
        if self.ndim > 1:
            raise ValueError('Dimension of calculation is greater than one')
        args = list(map(as_property, args))
        self._expand_property_arguments(args)

        arr_size = self.eq.GM.size
        results = dict()

        prop_MU_values = self.eq.MU
        conds_keys = [str(k) for k in self.eq.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
        local_index = 0

        for index, composition_sets in self.enumerate_composition_sets():
            cur_conds = OrderedDict(zip(conds_keys,
                                        [np.asarray(self.eq.coords[b][a], dtype=np.float_)
                                        for a, b in zip(index, conds_keys)]))
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
        x = as_property(x)
        data = self.get(x, *ys, values_only=False)
        
        for y in data.keys():
            if y == x:
                continue
            ax.plot(data[x], data[y], label=str(y))
        ax.legend()

# Upstream bug: Values are not cast before setattr is called, when frozen=False
def _setattr_workaround(self, name, value):
    try:
        field = self.__dataclass_fields__[name]
    except (KeyError, AttributeError):
        pass
    else:
        if (value is not None) and not isa(value, field.type):
            try:
                value = field.type.cast_from(value)
            except TypeError:
                pass
    object.__setattr__(self, name, value)
Workspace.__setattr__ = _setattr_workaround