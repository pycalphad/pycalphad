import pycalphad.variables as v
from pycalphad.codegen.sympydiff_utils import build_functions
from pycalphad.core.utils import get_pure_elements, unpack_species, \
    extract_parameters, get_state_variables
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.constraints import build_constraints
from pycalphad.core.errors import ConditionError
from itertools import repeat
from functools import lru_cache
import numpy as np

class PhaseRecordFactory(object):
    def __init__(self, dbf, comps, state_variables, models, parameters=None):
        self.comps = sorted(unpack_species(dbf, comps))
        self.pure_elements = get_pure_elements(dbf, comps)
        self.nonvacant_elements = sorted([x for x in self.pure_elements if x != 'VA'])
        self.molar_masses = np.array([dbf.refstates[x]['mass'] for x in self.nonvacant_elements], dtype='float')
        self._build_component_basis(dbf, comps)
        parameters = parameters if parameters is not None else {}
        self.models = models
        self.state_variables = sorted(get_state_variables(models=models, conds=state_variables), key=str)
        self.param_symbols, self.param_values = extract_parameters(parameters)

        if len(self.param_values.shape) > 1:
            self.param_values = self.param_values[0]

    def _build_component_basis(self, dbf, comps):
        """Build the change-of-basis matrix S (components x non-vacant elements).

        ``S[c, e]`` is the number of atoms of non-vacant element ``e`` per formula unit
        of component ``c``. A *redefined* (non-trivial) component basis must be exactly
        square and invertible: one component per non-vacant element, spanning the element
        space independently (e.g. AL2O3, ND2O3, ZRO2, O2 over {AL, ND, O, ZR}).

        For ordinary calculations the components reduce to the pure elements, giving the
        identity basis ("trivial"), in which case all downstream code uses the original
        element-basis paths unchanged. Normal component/species lists are frequently
        over-determined relative to the elements (charged species, or species lists
        expanded by ``unpack_species`` in ``calculate``); those are not redefined bases
        and fall back to the trivial element basis without error.
        """
        # Charged species (e.g. F and F-1.0) share the same neutral elemental composition
        # and are redundant for a basis over neutral elements; collapse them.
        components_by_composition = {}
        for c in v.unpack_components(comps, dbf):
            if c.number_of_atoms == 0:
                continue  # vacancies are not components of the basis
            composition_key = frozenset((el, amt) for el, amt in c.constituents.items() if el != 'VA')
            if composition_key not in components_by_composition:
                components_by_composition[composition_key] = c
        basis_components = sorted(components_by_composition.values(), key=str)
        nonvacant_elements = self.nonvacant_elements
        n_elements = len(nonvacant_elements)
        n_components = len(basis_components)
        element_index = {el: j for j, el in enumerate(nonvacant_elements)}

        def _use_trivial_basis():
            "Element basis: S = I. Preserves all existing element-basis behavior."
            self.basis_components = [v.Component(el, {el: 1.0}) for el in nonvacant_elements]
            self.basis_component_index = {el: j for j, el in enumerate(nonvacant_elements)}
            self.component_basis = np.eye(n_elements)
            self.component_basis_inv_T = np.eye(n_elements)
            self.component_molar_masses = np.asarray(self.molar_masses, dtype=float).copy()
            self.basis_is_trivial = True

        if n_components == 0 or n_components > n_elements:
            # Empty system, or an over-determined (normal/expanded/ionic) component list:
            # not a redefined basis.
            _use_trivial_basis()
            return
        if n_components < n_elements:
            raise ConditionError(
                f"Component basis is incomplete: {n_components} component(s) "
                f"{[str(c) for c in basis_components]} cannot span {n_elements} non-vacant "
                f"element(s) {nonvacant_elements}. Provide exactly one component per non-vacant "
                f"element to redefine the basis.")
        # Square: build S and require it to be invertible.
        S = np.zeros((n_elements, n_elements))
        for i, comp in enumerate(basis_components):
            for el, amount in comp.constituents.items():
                if el == 'VA':
                    continue
                S[i, element_index[el]] = amount
        if np.linalg.matrix_rank(S) < n_components:
            raise ConditionError(
                f"Component basis is linearly dependent (rank {np.linalg.matrix_rank(S)} < "
                f"{n_components}); components {[str(c) for c in basis_components]} do not "
                f"independently span the non-vacant element space {nonvacant_elements}.")
        if np.allclose(S, np.eye(n_elements)):
            _use_trivial_basis()
            return
        self.basis_components = basis_components
        self.basis_component_index = {str(c): i for i, c in enumerate(basis_components)}
        self.component_basis = S
        self.component_basis_inv_T = np.linalg.inv(S.T)
        self.component_molar_masses = S @ np.asarray(self.molar_masses, dtype=float)
        self.basis_is_trivial = False

    def update_parameters(self, parameters):
        new_param_symbols, new_param_values = extract_parameters(parameters)
        if len(new_param_values.shape) > 1:
            new_param_values = new_param_values[0]
        if new_param_symbols != self.param_symbols:
            raise ValueError('Parameter symbol mismatch')
        self.param_values[:] = new_param_values

    @lru_cache()
    def get_phase_constraints(self, phase_name):
        mod = self.models[phase_name]
        cfuncs = build_constraints(mod, self.state_variables + mod.site_fractions, parameters=self.param_symbols)
        return cfuncs

    @lru_cache()
    def get_phase_formula_moles_element(self, phase_name, element_name, per_formula_unit=True):
        mod = self.models[phase_name]
        # TODO: In principle, we should also check for undefs in mod.moles()
        return build_functions(mod.moles(element_name, per_formula_unit=per_formula_unit),
                               self.state_variables + mod.site_fractions,
                               include_obj=True, include_grad=True, include_hess=True,
                               parameters=self.param_symbols)

    @lru_cache()
    def get_phase_property(self, phase_name, property_name, include_grad=True, include_hess=True):
        mod = self.models[phase_name]
        out = getattr(mod, property_name)
        if out is None:
            raise AttributeError(f'Model property {property_name} is not defined')
        # Only force undefineds to zero if we're not overriding them
        undefs = {x for x in out.free_symbols if not isinstance(x, v.StateVariable)} - set(self.param_symbols)
        undef_vals = repeat(0., len(undefs))
        out = out.xreplace(dict(zip(undefs, undef_vals)))
        build_output = build_functions(out, tuple(self.state_variables + mod.site_fractions), parameters=self.param_symbols,
                                       include_grad=include_grad, include_hess=include_hess)
        return build_output

    def get_phase_formula_energy(self, phase_name):
        return self.get_phase_property(phase_name, 'G', include_grad=True, include_hess=True)

    @lru_cache()
    def get(self, phase_name):
        return PhaseRecord(self, phase_name)

    def keys(self):
        return self.models.keys()

    def values(self):
        return iter(self.get(k) for k in self.keys())

    def items(self):
        return zip(self.models.keys(), iter(self.get(k) for k in self.keys()))

    __getitem__ = get
