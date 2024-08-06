import warnings
from collections import OrderedDict, Counter, defaultdict
from copy import copy
from pycalphad.property_framework.computed_property import JanssonDerivative
import pycalphad.variables as v
from pycalphad.core.utils import unpack_components, unpack_condition, unpack_phases, filter_phases, instantiate_models
from pycalphad import calculate
from pycalphad.core.starting_point import starting_point
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.eqsolver import _solve_eq_at_conditions
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.solver import Solver, SolverBase
from pycalphad.core.light_dataset import LightDataset
from pycalphad.model import Model
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, Type
from pycalphad.io.database import Database
from pycalphad.variables import Species, StateVariable
from pycalphad.core.conditions import Conditions, ConditionError
from pycalphad.property_framework import ComputableProperty, as_property
from pycalphad.property_framework.units import unit_conversion_context, ureg, as_quantity, Q_
from runtype import isa
from runtype.pytypes import Dict, List, Sequence, SumType, Mapping, NoneType
from typing import TypeVar



def _adjust_conditions(conds) -> OrderedDict[StateVariable, List[float]]:
    "Adjust conditions values to be in the implementation units of the quantity, and within the numerical limit of the solver."
    new_conds = OrderedDict()
    minimum_composition = 1e-10
    for key, value in sorted(conds.items(), key=str):
        key = as_property(key)
        # If conditions have units, convert to impl units and strip them
        if isinstance(value, Q_):
            value = value.to(key.implementation_units).magnitude
        if isinstance(key, v.MoleFraction):
            vals = unpack_condition(value)
            # "Zero" composition is a common pattern. Do not warn for that case.
            if np.any(np.logical_and(np.asarray(vals) < minimum_composition, np.asarray(vals) > 0)):
                warnings.warn(
                    f"Some specified compositions are below the minimum allowed composition of {minimum_composition}.")
            new_conds[key] = [min(max(val, minimum_composition), 1-minimum_composition) for val in vals]
        else:
            new_conds[key] = unpack_condition(value)
        if getattr(key, 'display_units', '') != '':
            new_conds[key] = Q_(new_conds[key], units=key.display_units).to(key.implementation_units)
    return new_conds

class SpeciesList:
    @classmethod
    def cast_from(cls, s: Sequence) -> "SpeciesList":
        return sorted(Species.cast_from(x) for x in s)

class PhaseList:
    @classmethod
    def cast_from(cls, s: SumType([str, Sequence[str]])) -> "PhaseList":
        if isinstance(s, str):
            s = [s]
        return sorted(PhaseName.cast_from(x) for x in s)

class PhaseName:
    @classmethod
    def cast_from(cls, s: str) -> "PhaseName":
        return s.upper()

class ConditionValue:
    @classmethod
    def cast_from(cls, value: SumType([float, Sequence[float]])) -> "ConditionValue":
        return unpack_condition(value)

class ConditionKey:
    @classmethod
    def cast_from(cls, key: SumType([str, StateVariable])) -> "ConditionKey":
        return as_property(key)

class TypedField:
    """
    A descriptor for managing attributes with specific types in a class, supporting automatic type coercion and default values.
    This class is designed to be used in scenarios (like `Workspace`) where one needs to implement an observer pattern. It enables the tracking of changes in attribute values and notifies dependent attributes of any updates.
    """

    def __init__(self, default_factory=None, depends_on=None):
        """
        Attributes
        ----------
        default_factory : callable, optional
            A callable that returns the default value of the attribute when no initial value is provided.
        depends_on : list of str, optional
            A list of attribute names, from the parent object, that the current attribute depends on. Changes to these attributes will trigger updates to the current attribute.
        """
        self.default_factory = default_factory
        self.depends_on = depends_on

    def __set_name__(self, owner, name):
        "Initializes the attribute, determining its private and public names and registering dependency callbacks if necessary."
        self.type = owner.__annotations__.get(name, None)
        self.public_name = name
        self.private_name = '_' + name
        if self.depends_on is not None:
            for dependency in self.depends_on:
                owner._callbacks[dependency].append(self.on_dependency_update)

    def __set__(self, obj, value):
        "Sets the value of the attribute in an object, handling type coercion via the `cast_from` method if the direct assignment isn't possible. It raises `TypeError` if coercion fails."
        if (self.type != NoneType) and not isa(value, self.type) and value is not None:
            value = self.type.cast_from(value)
        elif value is None and self.default_factory is not None:
            value = self.default_factory(obj)
        oldval = getattr(obj, self.private_name, None)
        setattr(obj, self.private_name, value)
        for cb in obj._callbacks[self.public_name]:
            cb(obj, self.public_name, oldval, value)

    def __get__(self, obj, objtype=None):
        "Retrieves the value of the attribute, initializing it with default_factory if it hasn't been set before."
        if not hasattr(obj, self.private_name):
            if self.default_factory is not None:
                default_value = self.default_factory(obj)
                setattr(obj, self.private_name, default_value)
        return getattr(obj, self.private_name)

    def on_dependency_update(self, obj, updated_attribute, old_val, new_val):
        "A callback method that can be overridden to define custom behavior when a dependent attribute is updated."
        pass

class ComponentsField(TypedField):
    def __init__(self, depends_on=None):
        super().__init__(default_factory=lambda obj: unpack_components(obj.database, sorted(x.name for x in obj.database.species if x.name != '/-')),
                         depends_on=depends_on)
    def __set__(self, obj, value):
        comps = sorted(unpack_components(obj.database, value))
        super().__set__(obj, comps)

    def __get__(self, obj, objtype=None):
        getobj = super().__get__(obj, objtype=objtype)
        return sorted(unpack_components(obj.database, getobj))

class PhasesField(TypedField):
    def __init__(self, depends_on=None):
        super().__init__(default_factory=lambda obj: filter_phases(obj.database, obj.components),
                         depends_on=depends_on)
    def __set__(self, obj, value):
        phases = sorted(unpack_phases(value))
        super().__set__(obj, phases)

    def __get__(self, obj, objtype=None):
        getobj = super().__get__(obj, objtype=objtype)
        return filter_phases(obj.database, obj.components, getobj)

class DictField(TypedField):
    def get_proxy(self, obj):
        class DictProxy:
            @staticmethod
            def unwrap():
                return TypedField.__get__(self, obj)
            def __getattr__(pxy, name):
                getobj = TypedField.__get__(self, obj)
                if getobj == pxy:
                    raise ValueError('Proxy object points to itself')
                return getattr(getobj, name)
            def __getitem__(pxy, item):
                return TypedField.__get__(self, obj).get(item)
            def __iter__(pxy):
                return TypedField.__get__(self, obj).__iter__()
            def __setitem__(pxy, item, value):
                conds = TypedField.__get__(self, obj)
                conds[item] = value
                self.__set__(obj, conds)
            def update(pxy, new_conds):
                conds = TypedField.__get__(self, obj)
                conds.update(new_conds)
                self.__set__(obj, conds)
            def __delitem__(pxy, item):
                conds = TypedField.__get__(self, obj)
                del conds[item]
                self.__set__(obj, conds)
            def __len__(pxy):
                return len(TypedField.__get__(self, obj))
            def __str__(pxy):
                return str(TypedField.__get__(self, obj))
            def __repr__(pxy):
                return repr(TypedField.__get__(self, obj))
        return DictProxy()

    def __get__(self, obj, objtype=None):
        return self.get_proxy(obj)

class ConditionsField(DictField):
    def __set__(self, obj, value):
        conds = Conditions(obj)
        for k, v in value.items():
            conds[k] = v
        super().__set__(obj, conds)

class ModelsField(DictField):
    def __init__(self, depends_on=None):
        super().__init__(default_factory=lambda obj: instantiate_models(obj.database, obj.components, obj.phases,
                                                                        model=None, parameters=obj.parameters),
                         depends_on=depends_on)
    def __set__(self, obj, value):
        # Unwrap proxy objects before being stored
        if hasattr(value, 'unwrap'):
            value = value.unwrap()
        try:
            # Expand specified Model type into a dict of instances
            value = instantiate_models(obj.database, obj.components, obj.phases, model=value, parameters=obj.parameters)
            super().__set__(obj, value)
        except AttributeError:
            super().__set__(obj, None)

    def on_dependency_update(self, obj, updated_attribute, old_val, new_val):
        self.__set__(obj, self.default_factory(obj))

class PRFField(TypedField):
    def __init__(self, depends_on=None):
        def make_prf(obj):
            try:
                prf = PhaseRecordFactory(obj.database, obj.components, obj.conditions,
                                         obj.models.unwrap() if hasattr(obj.models, 'unwrap') else obj.models,
                                         parameters=obj.parameters)
                return prf
            except AttributeError:
                return None
        super().__init__(default_factory=make_prf, depends_on=depends_on)

    def on_dependency_update(self, obj, updated_attribute, old_val, new_val):
        self.__set__(obj, self.default_factory(obj))

class SolverField(TypedField):
    def on_dependency_update(self, obj, updated_attribute, old_val, new_val):
        self.__set__(obj, self.default_factory(obj))

class EquilibriumCalculationField(TypedField):
    def __get__(self, obj, objtype=None):
        if (not hasattr(obj, self.private_name)) or (getattr(obj, self.private_name) is None):
            try:
                default_value = obj.recompute()
            except AttributeError:
                default_value = None
            setattr(obj, self.private_name, default_value)
        return getattr(obj, self.private_name)

    def on_dependency_update(self, obj, updated_attribute, old_val, new_val):
        self.__set__(obj, None)


# Defined to allow type checking for Model or its subclasses
ModelType = TypeVar('ModelType', bound=Model)

class Workspace:
    _callbacks = defaultdict(lambda: [])
    database: Database = TypedField(lambda _: None)
    components: SpeciesList = ComponentsField(depends_on=['database'])
    phases: PhaseList = PhasesField(depends_on=['database', 'components'])
    conditions: Conditions = ConditionsField(lambda wks: Conditions(wks))
    verbose: bool = TypedField(lambda _: False)
    models: Mapping[PhaseName, ModelType] = ModelsField(depends_on=['phases', 'parameters'])
    parameters: SumType([NoneType, Dict]) = DictField(lambda _: OrderedDict())
    phase_record_factory: Optional[PhaseRecordFactory] = PRFField(depends_on=['phases', 'conditions', 'models', 'parameters'])
    calc_opts: SumType([NoneType, Dict]) = DictField(lambda _: OrderedDict())
    solver: SolverBase = SolverField(lambda obj: Solver(verbose=obj.verbose), depends_on=['verbose'])
    # eq is set by a callback in the EquilibriumCalculationField (TypedField)
    eq: Optional[LightDataset] = EquilibriumCalculationField(depends_on=['phase_record_factory', 'calc_opts', 'solver'])

    def __init__(self, *args, **kwargs):
        # Assume positional arguments are specified in class typed-attribute definition order
        for arg, attrname in zip(args, ['database', 'components', 'phases', 'conditions']):
            setattr(self, attrname, arg)
        attributes = list(self.__annotations__.keys())
        for kwarg_name, kwarg_val in kwargs.items():
            if kwarg_name not in attributes:
                raise ValueError(f'{kwarg_name} is not a Workspace attribute')
            setattr(self, kwarg_name, kwarg_val)

    def recompute(self):
        # Assumes implementation units from this point
        unitless_conds = OrderedDict((key, as_quantity(key, value).to(key.implementation_units).magnitude) for key, value in self.conditions.items())
        str_conds = OrderedDict((str(key), value) for key, value in unitless_conds.items())
        local_conds = {key: as_quantity(key, value).to(key.implementation_units).magnitude
                       for key, value in self.conditions.items()
                       if getattr(key, 'phase_name', None) is not None}
        state_variables = self.phase_record_factory.state_variables
        self.phase_record_factory.update_parameters(self.parameters.unwrap())

        # 'calculate' accepts conditions through its keyword arguments
        grid_opts = self.calc_opts.copy()
        statevar_strings = [str(x) for x in state_variables]
        grid_opts.update({key: value for key, value in str_conds.items() if key in statevar_strings})

        if 'pdens' not in grid_opts:
            grid_opts['pdens'] = 60

        grid = calculate(self.database, self.components, self.phases, model=self.models.unwrap(), fake_points=True,
                        phase_records=self.phase_record_factory, output='GM', parameters=self.parameters.unwrap(),
                        to_xarray=False, conditions=local_conds, **grid_opts)
        properties = starting_point(unitless_conds, state_variables, self.phase_record_factory, grid)
        return _solve_eq_at_conditions(properties, self.phase_record_factory, grid,
                                       list(unitless_conds.keys()), state_variables,
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
            elif hasattr(args[i], 'sublattice_index') and args[i].sublattice_index == '*':
                # We need to resolve sublattice_index before species to ensure we
                # get the correct set of phase constituents for each sublattice
                indices_to_delete.append(i)
                sublattice_indices = sorted(set([x.sublattice_index for x in self.phase_record_factory[args[i].phase_name].variables]))
                additional_args = args[i].expand_wildcard(sublattice_indices=sublattice_indices)
                args.extend(additional_args)
            elif hasattr(args[i], 'species') and args[i].species == v.Species('*'):
                indices_to_delete.append(i)
                internal_to_phase = hasattr(args[i], 'sublattice_index')
                if internal_to_phase:
                    components = [x.species for x in self.phase_record_factory[args[i].phase_name].variables
                                  if x.sublattice_index == args[i].sublattice_index]
                else:
                    # TODO: self.components with proper Components support
                    components = [comp for comp in self.phase_record_factory.nonvacant_elements]
                additional_args = args[i].expand_wildcard(components=components)
                args.extend(additional_args)
            elif isinstance(args[i], JanssonDerivative):
                numerator_args = [args[i].numerator]
                self._expand_property_arguments(numerator_args)
                denominator_args = [args[i].denominator]
                self._expand_property_arguments(denominator_args)
                if (len(numerator_args) > 1) or (len(denominator_args) > 1):
                    for n_arg in numerator_args:
                        for d_arg in denominator_args:
                            args.append(JanssonDerivative(n_arg, d_arg))
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

    @property
    def ndim(self) -> int:
        _ndim = 0
        for cond_val in self.conditions.values():
            if len(cond_val) > 1:
                _ndim += 1
        return _ndim

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
                                        [np.asarray(self.eq.coords[b][a], dtype=np.float64)
                                        for a, b in zip(index, conds_keys)]))
            state_variable_values = [cur_conds[key] for key in str_state_variables]
            state_variable_values = np.array(state_variable_values)
            composition_sets = []
            for phase_idx, phase_name in enumerate(prop_Phase_values[index]):
                if phase_name == '' or phase_name == '_FAKE_':
                    continue
                # phase_name can be a numpy.str_, which is different from the builtin str
                phase_record = self.phase_record_factory[str(phase_name)]
                sfx = prop_Y_values[index + np.index_exp[phase_idx, :phase_record.phase_dof]]
                phase_amt = prop_NP_values[index + np.index_exp[phase_idx]]
                compset = CompositionSet(phase_record)
                compset.update(sfx, phase_amt, state_variable_values)
                composition_sets.append(compset)
            yield index, composition_sets

    def get_composition_sets(self):
        if self.ndim != 0:
            raise ConditionError('get_composition_sets() can only be used for point (0-D) calculations. Use enumerate_composition_sets() instead.')
        return next(self.enumerate_composition_sets())[1]

    @property
    def condition_axis_order(self):
        str_conds_keys = [str(k) for k in self.eq.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
        conds_keys = [None] * len(str_conds_keys)
        for k in self.conditions.keys():
            cond_idx = str_conds_keys.index(str(k))
            # unit-length dimensions will be 'squeezed' out
            if len(self.eq.coords[str(k)]) > 1:
                conds_keys[cond_idx] = k
        return [c for c in conds_keys if c is not None]

    def get_dict(self, *args: Tuple[ComputableProperty]):
        args = list(map(as_property, args))
        self._expand_property_arguments(args)
        arg_units = {arg: (ureg.Unit(getattr(arg, 'implementation_units', '')),
                           ureg.Unit(getattr(arg, 'display_units', '')))
                     for arg in args}

        arr_size = self.eq.GM.size
        results = dict()

        prop_MU_values = self.eq.MU
        str_conds_keys = [str(k) for k in self.eq.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
        conds_keys = [None] * len(str_conds_keys)
        for k in self.conditions.keys():
            cond_idx = str_conds_keys.index(str(k))
            conds_keys[cond_idx] = k
        local_index = 0

        for index, composition_sets in self.enumerate_composition_sets():
            cur_conds = OrderedDict(zip(conds_keys,
                                        [np.asarray(self.eq.coords[b][a], dtype=np.float64)
                                        for a, b in zip(index, str_conds_keys)]))
            chemical_potentials = prop_MU_values[index]

            for arg in args:
                prop_implementation_units, prop_display_units = arg_units[arg]
                context = unit_conversion_context(composition_sets, arg)
                if results.get(arg, None) is None:
                    results[arg] = np.zeros((arr_size,) + arg.shape)
                results[arg][local_index, ...] = Q_(arg.compute_property(composition_sets, cur_conds, chemical_potentials),
                                                    prop_implementation_units).to(prop_display_units, context).magnitude
            local_index += 1

        # roll the dimensions of the property arrays back up
        conds_shape = tuple(len(self.eq.coords[str(b)]) for b in self.condition_axis_order)
        for arg in results.keys():
            results[arg] = results[arg].reshape(conds_shape + arg.shape)

        return results

    def get(self, *args: Tuple[ComputableProperty]):
        result = list(self.get_dict(*args).values())
        if len(result) != 1:
            return result
        else:
            # For single properties, just return the result without wrapping in a list
            return result[0]

    def copy(self):
        return copy(self)
