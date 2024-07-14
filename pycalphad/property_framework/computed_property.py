import numpy.typing as npt
import numpy as np
from typing import Dict, Union, List, Optional
from symengine import Basic, Mul, Pow, S
import pycalphad.variables as v
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.solver import Solver
from pycalphad.property_framework.types import ComputableProperty, JanssonDerivativeDeltas, \
    DifferentiableComputableProperty, ConditionableComputableProperty
from pycalphad.property_framework import units
from copy import copy

class ModelComputedProperty(object):
    def __init__(self, model_attr_name: str, phase_name: Optional[str] = None):
        self.implementation_units = getattr(units, model_attr_name + '_implementation_units', '')
        self.display_units = getattr(units, model_attr_name + '_display_units', '')
        self.display_name = getattr(units, model_attr_name + '_display_name', model_attr_name)
        self.model_attr_name = model_attr_name
        self.phase_name = phase_name

    def __getitem__(self, new_units: str) -> "ModelComputedProperty":
        "Get ModelComputedProperty with different display units"
        newobj = copy(self)
        newobj.display_units = new_units
        return newobj

    def expand_wildcard(self, phase_names):
        return [self.__class__(self.model_attr_name, phase_name) for phase_name in phase_names]

    def __str__(self):
        result = self.model_attr_name
        if self.phase_name is not None:
            result += f'({self.phase_name})'
        return result

    def __eq__(self, other):
        if self is other:
            return True
        if self.__class__ != other.__class__:
            return False
        if self.__dict__ == other.__dict__:
            return True
        return False

    def __hash__(self):
        return hash(str(self))

    @property
    def shape(self):
        "Shape of return value is a scalar."
        return tuple()

    @property
    def multiplicity(self):
        "Indicates multiplicity of a composition set, i.e., returns `2` for property of `FCC_A1#2` phase."
        if self.phase_name is not None:
            tokens = self.phase_name.split('#')
            if len(tokens) > 1:
                return int(tokens[1])
            else:
                return 1
        else:
            return None

    def compute_property(self, compsets: List[CompositionSet], cur_conds: Dict[str, float], chemical_potentials: npt.ArrayLike) -> npt.ArrayLike:
        if len(compsets) == 0:
            return np.nan
        if self.phase_name is None:
            return np.sum([compset.NP*self._compute_per_phase_property(compset, cur_conds) for compset in compsets])
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
                    return self._compute_per_phase_property(compset, cur_conds)
            return np.nan


    def jansson_derivative(self, compsets, cur_conds, chemical_potentials, deltas: JanssonDerivativeDeltas) -> npt.ArrayLike:
        "Compute Jansson derivative with self as numerator, with the given deltas"
        state_variables = compsets[0].phase_record.state_variables
        grad_values = self._compute_property_gradient(compsets, cur_conds, chemical_potentials)

        # Sundman et al, 2015, Eq. 73
        jansson_derivative = np.nan
        for idx, compset in enumerate(compsets):
            if compset.NP == 0 and not (compset.fixed):
                continue
            func_value = self._compute_per_phase_property(compset, cur_conds)
            if np.isnan(func_value):
                continue
            if np.isnan(jansson_derivative):
                jansson_derivative = 0.0
            grad_value = grad_values[idx]
            delta_sitefracs = deltas.delta_sitefracs[idx]

            if self.phase_name is None:
                jansson_derivative += deltas.delta_phase_amounts[idx] * func_value
                jansson_derivative += compset.NP * np.dot(deltas.delta_statevars, grad_value[:len(state_variables)])
                jansson_derivative += compset.NP * np.dot(delta_sitefracs, grad_value[len(state_variables):])
            else:
                jansson_derivative += np.dot(deltas.delta_statevars, grad_value[:len(state_variables)])
                jansson_derivative += np.dot(delta_sitefracs, grad_value[len(state_variables):])
        return jansson_derivative

    def _compute_per_phase_property(self, compset: CompositionSet, cur_conds: Dict[str, float]) -> float:
        out = np.atleast_1d(np.zeros(1))
        compset.phase_record.prop(out, compset.dof, self.model_attr_name.encode('utf-8'))
        return out[0]

    def _compute_property_gradient(self, compsets, cur_conds, chemical_potentials):
        "Compute partial derivatives of property with respect to degrees of freedom of given CompositionSets"
        if self.phase_name is not None:
            tokens = self.phase_name.split('#')
            phase_name = tokens[0]
        result = [np.zeros(compset.dof.shape[0]) for compset in compsets]
        multiplicity_seen = 0
        for cs_idx, compset in enumerate(compsets):
            if (self.phase_name is not None) and (compset.phase_record.phase_name != phase_name):
                continue
            if self.multiplicity is not None:
                multiplicity_seen += 1
                if self.multiplicity != multiplicity_seen:
                    continue
            grad = np.zeros((compset.dof.shape[0]))
            compset.phase_record.prop_grad(grad, compset.dof, self.model_attr_name.encode('utf-8'))
            result[cs_idx][:] = grad
        return result

class LinearCombination:
    display_units = ''
    implementation_units = ''
    def __init__(self, expr: Basic):
        symbols = sorted(expr.free_symbols, key=str)
        symbol_classes = {s.__class__ for s in symbols}
        if len(symbol_classes) > 1:
            raise ValueError(f'Property types in a linear combination must match. Got: {expr}')
        if list(symbol_classes)[0] != v.MoleFraction:
            raise ValueError('Only mole fractions are supported in linear combination conditions')
        # Detect case of molar ratio (x/y = c); convert to (x - c*y = 0)
        denominator = S.One
        for mul_atom in expr.atoms(Mul):
            # Division is stored as a Mul where one argument is a reciprocal
            for mul_arg in mul_atom.args:
                if isinstance(mul_arg, Pow) and isinstance(mul_arg.args[0], v.StateVariable):
                    denominator = mul_arg.args[0]
        expr = (expr*denominator).expand()
        coefs = []
        for s in symbols:
            coef = expr.diff(s)
            if float(coef) == int(coef):
                coef = int(coef)
            else:
                coef = float(coef)
            coefs.append(coef)
        constant_term = expr + 0
        for symbol, coef in zip(symbols, coefs):
            constant_term -= symbol*coef
        constant_term = float(constant_term)
        symbols.append(S.One)
        coefs.append(constant_term)
        self.coefs = coefs
        self.symbols = symbols
        self.denominator = denominator

    def __str__(self):
        return f"LinComb_{'-'.join([str(s) for s in self.symbols])},{'-'.join([str(s) for s in self.coefs])}"

    def __repr__(self):
        result = ""
        for idx, (sym, coef) in enumerate(zip(self.symbols[:-1], self.coefs[:-1])):
            result += str(coef) + '*' + repr(sym)
            if idx + 1 < len(self.symbols):
                # if not the last entry
                result += '+'
        result += '=' + str(self.coefs[-1])
        return result

    @property
    def shape(self):
        return tuple()

    def compute_property(self, compsets: List[CompositionSet], cur_conds: Dict[str, float], chemical_potentials: npt.ArrayLike) -> npt.ArrayLike:
        result = 0.0
        for symbol, coef in zip(self.symbols, self.coefs):
            if symbol == S.One:
                result += coef
            else:
                sym_val = symbol.compute_property(compsets, cur_conds, chemical_potentials)
                result += coef*sym_val
        return result

def as_property(inp: Union[str, Basic, ComputableProperty]) -> ComputableProperty:
    if isinstance(inp, ComputableProperty):
        return inp
    elif isinstance(inp, Basic):
        # Try to convert mathematical expression to a LinComb condition
        inp = LinearCombination(inp)
        return inp
    elif not isinstance(inp, str):
        raise TypeError(f'{inp} is not a ComputableProperty')
    dot_tokens = inp.split('.')
    if len(dot_tokens) == 2:
        numerator, denominator = dot_tokens
        numerator = as_property(numerator)
        denominator = as_property(denominator)
        return JanssonDerivative(numerator, denominator)
    try:
        begin_parens = inp.index('(')
        end_parens = inp.index(')')
    except ValueError:
        begin_parens = len(inp)
        end_parens = len(inp)

    specified_prop = inp[:begin_parens].strip()

    prop = getattr(v, specified_prop, None)
    if prop is None:
        prop = ModelComputedProperty
    if begin_parens != end_parens:
        specified_args = tuple(x.strip() for x in inp[begin_parens+1:end_parens].split(','))
        if not isinstance(prop, type):
            prop_instance = type(prop)(*((specified_prop,)+specified_args))
        else:
            if issubclass(prop, v.StateVariable):
                prop_instance = prop(*(specified_args))
            else:
                prop_instance = prop(*((specified_prop,)+specified_args))
    else:
        if isinstance(prop, type):
            prop = prop(specified_prop)
        prop_instance = prop
    return prop_instance

class JanssonDerivative:
    def __init__(self, numerator: DifferentiableComputableProperty, denominator: ConditionableComputableProperty):
        self.numerator = as_property(numerator)
        if not isinstance(self.numerator, DifferentiableComputableProperty):
            raise TypeError(f'{self.numerator} is not a differentiable property')
        self.denominator = as_property(denominator)
        if not isinstance(self.denominator, ConditionableComputableProperty):
            raise TypeError(f'{self.denominator} cannot be used in the denominator of a Jansson derivative')

    @property
    def shape(self):
        return tuple()

    @property
    def implementation_units(self):
        return str(units.ureg.Unit(self.numerator.implementation_units) / units.ureg.Unit(self.denominator.implementation_units))

    _display_units = None
    @property
    def display_units(self):
        if self._display_units is not None:
            return self._display_units
        else:
            return str(units.ureg.Unit(self.numerator.display_units) / units.ureg.Unit(self.denominator.display_units))

    @display_units.setter
    def display_units(self, val):
        self._display_units = val

    def __getitem__(self, new_units: str) -> "JanssonDerivative":
        "Get JanssonDerivative with different display units"
        newobj = copy(self)
        newobj.display_units = new_units
        return newobj

    _display_name = None
    @property
    def display_name(self):
        if self._display_name is not None:
            return self._display_name
        else:
            return str(self)

    @display_name.setter
    def display_name(self, val):
        self._display_name = val

    def compute_property(self, compsets, cur_conds, chemical_potentials):
        if len(compsets) == 0:
            return np.nan
        solver = Solver()
        spec = solver.get_system_spec(compsets, cur_conds)
        state = spec.get_new_state(compsets)
        state.chemical_potentials[:] = chemical_potentials
        state.recompute(spec)
        deltas = self.denominator.jansson_deltas(spec, state)
        return self.numerator.jansson_derivative(compsets, cur_conds, chemical_potentials, deltas)

    def __str__(self):
        return str(self.numerator)+'.'+str(self.denominator)

    def __eq__(self, other):
        if self is other:
            return True
        if self.__class__ != other.__class__:
            return False
        if self.__dict__ == other.__dict__:
            return True
        return False

    def __hash__(self):
        return hash(str(self))