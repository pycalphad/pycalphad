import pycalphad.variables as v
from pycalphad.codegen.sympydiff_utils import build_functions
from pycalphad.core.utils import get_pure_elements, unpack_components, \
    extract_parameters, get_state_variables
from pycalphad.core.phase_rec import PhaseRecord
from pycalphad.core.constraints import build_constraints
from itertools import repeat
import warnings
from functools import lru_cache


def build_phase_records(dbf, comps, phases, state_variables, models, output='GM',
                        callables=None, parameters=None, verbose=False,
                        build_gradients=True, build_hessians=True
                        ):
    if output != 'GM':
        raise ValueError('build_phase_records is deprecated and no longer works when the output keyword '
                         'is changed from the default. Remove the keyword, and then use the PhaseRecord.prop_* API '
                         'in downstream functions instead.')
    return PhaseRecordFactory(dbf, comps, state_variables, models, parameters=parameters)

class PhaseRecordFactory(object):
    def __init__(self, dbf, comps, state_variables, models, parameters=None):
        self.comps = sorted(unpack_components(dbf, comps))
        self.pure_elements = get_pure_elements(dbf, comps)
        self.nonvacant_elements = sorted([x for x in self.pure_elements if x != 'VA'])
        parameters = parameters if parameters is not None else {}
        self.models = models
        self.state_variables = sorted(get_state_variables(models=models, conds=state_variables), key=str)
        print(self.state_variables)
        self.param_symbols, self.param_values = extract_parameters(parameters)

        if len(self.param_values.shape) > 1:
            self.param_values = self.param_values[0]

    @lru_cache
    def get_phase_constraints(self, phase_name):
        mod = self.models[phase_name]
        cfuncs = build_constraints(mod, self.state_variables + mod.site_fractions, parameters=self.param_symbols)
        return cfuncs

    @lru_cache
    def get_phase_formula_moles_element(self, phase_name, element_name, per_formula_unit=True):
        mod = self.models[phase_name]
        # TODO: In principle, we should also check for undefs in mod.moles()
        return build_functions(mod.moles(element_name, per_formula_unit=per_formula_unit),
                               self.state_variables + mod.site_fractions,
                               include_obj=True, include_grad=True, include_hess=True,
                               parameters=self.param_symbols)

    @lru_cache
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

    @lru_cache
    def get(self, phase_name):
        return PhaseRecord(self, phase_name)

    def keys(self):
        return self.models.keys()

    def values(self):
        return iter(self.get(k) for k in self.keys())

    __getitem__ = get
