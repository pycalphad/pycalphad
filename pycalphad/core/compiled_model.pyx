from sympy import Add, Float, Integer, Rational, Mul, Pow, S, collect, Symbol, \
    Piecewise, Intersection, EmptySet, Union, Interval, log
from tinydb import where
import pycalphad.variables as v
from pycalphad.io.tdb import to_interval
from pycalphad import Model
import copy
import numpy as np
cimport numpy as np

# Maximum number of levels deep we check for symbols that are functions of
# other symbols
_MAX_PARAM_NESTING = 32

def build_piecewise_matrix(sympy_obj, cur_exponents, low_temp, high_temp, output_matrix, symbol_matrix, param_symbols):
    if isinstance(sympy_obj, (Float, Integer, Rational)):
        result = float(sympy_obj)
        if result != 0.0:
            output_matrix.append([low_temp, high_temp] + list(cur_exponents) + [result])
    elif isinstance(sympy_obj, Piecewise):
        piece_args = [i for i in sympy_obj.args if i.expr != S.Zero]
        intervals = [to_interval(i.cond) for i in piece_args]
        if (len(intervals) > 1) and Intersection(intervals) != EmptySet():
            raise ValueError('Overlapping intervals cannot be represented: {}'.format(intervals))
        if not isinstance(Union(intervals), Interval):
            raise ValueError('Piecewise intervals must be continuous')
        if not all([arg.cond.free_symbols == {v.T} for arg in piece_args]):
            raise ValueError('Only temperature-dependent piecewise conditions are supported')
        exprs = [arg.expr for arg in piece_args]
        for expr, invl in zip(exprs, intervals):
            lowlim = invl.args[0] if invl.args[0] > low_temp else low_temp
            highlim = invl.args[1] if invl.args[1] < high_temp else high_temp
            if highlim < lowlim:
                continue
            build_piecewise_matrix(expr, cur_exponents, float(lowlim), float(highlim), output_matrix, symbol_matrix, param_symbols)
    elif isinstance(sympy_obj, Symbol):
        symbol_matrix.append([low_temp, high_temp] + list(cur_exponents) + [param_symbols.index(sympy_obj)])
    elif isinstance(sympy_obj, Add):
        sympy_obj = sympy_obj.expand()
        for arg in sympy_obj.args:
            build_piecewise_matrix(arg, cur_exponents, low_temp, high_temp, output_matrix, symbol_matrix, param_symbols)
    elif isinstance(sympy_obj, Mul):
        new_exponents = np.array(cur_exponents)
        remaining_argument = S.One
        if (len(sympy_obj.args) == 2) and isinstance(sympy_obj.args[0], (Float, Integer, Rational)) and isinstance(sympy_obj.args[1], Piecewise):
            remaining_argument = Piecewise(*[(sympy_obj.args[0]*expr, cond) for expr, cond in sympy_obj.args[1].args])
        elif (len(sympy_obj.args) == 2) and isinstance(sympy_obj.args[0], Piecewise) and isinstance(sympy_obj.args[1], (Float, Integer, Rational)):
            remaining_argument = Piecewise(*[(sympy_obj.args[1]*expr, cond) for expr, cond in sympy_obj.args[0].args])
        else:
            for arg in sympy_obj.args:
                if isinstance(arg, Pow):
                    if arg.base == v.T:
                        new_exponents[1] += int(arg.exp)
                    elif arg.base == v.P:
                        new_exponents[0] += int(arg.exp)
                    else:
                        raise NotImplementedError
                elif arg == v.T:
                    new_exponents[1] += 1
                elif arg == v.P:
                    new_exponents[0] += 1
                elif arg == log(v.T):
                    new_exponents[3] += 1
                elif arg == log(v.P):
                    new_exponents[2] += 1
                else:
                    remaining_argument *= arg
        if not isinstance(remaining_argument, Mul):
            build_piecewise_matrix(remaining_argument, new_exponents, low_temp, high_temp,
                                   output_matrix, symbol_matrix, param_symbols)
        else:
            raise NotImplementedError(sympy_obj, type(sympy_obj))
    else:
        raise NotImplementedError


class RedlichKisterSum(object):
    def __init__(self, comps, phase, param_search, param_query, param_symbols, all_symbols, variable_rename_dict=None):
        """
        Construct parameter in Redlich-Kister polynomial basis, using
        the Muggianu ternary parameter extension.
        """
        variable_rename_dict = variable_rename_dict if variable_rename_dict is not None else dict()
        rk_terms = []
        dof = [v.SiteFraction(phase.name, subl_index, comp)
               for subl_index, subl in enumerate(phase.constituents) for comp in sorted(set(subl).intersection(comps))]
        self.output_matrix = []
        self.symbol_matrix = []

        # search for desired parameters
        params = param_search(param_query)
        for param in params:
            # iterate over every sublattice
            mixing_term = S.One
            for subl_index, comps in enumerate(param['constituent_array']):
                comp_symbols = None
                # convert strings to symbols
                if comps[0] == '*':
                    # Handle wildcards in constituent array
                    comp_symbols = \
                        [
                            v.SiteFraction(phase.name, subl_index, comp)
                            for comp in set(phase.constituents[subl_index])\
                                .intersection(self.components)
                        ]
                    mixing_term *= Add(*comp_symbols)
                else:
                    comp_symbols = \
                        [
                            v.SiteFraction(phase.name, subl_index, comp)
                            for comp in comps
                        ]
                    mixing_term *= Mul(*comp_symbols)
                # is this a higher-order interaction parameter?
                if len(comps) == 2 and param['parameter_order'] > 0:
                    # interacting sublattice, add the interaction polynomial
                    mixing_term *= Pow(comp_symbols[0] - \
                        comp_symbols[1], param['parameter_order'])
                if len(comps) == 3:
                    # 'parameter_order' is an index to a variable when
                    # we are in the ternary interaction parameter case

                    # NOTE: The commercial software packages seem to have
                    # a "feature" where, if only the zeroth
                    # parameter_order term of a ternary parameter is specified,
                    # the other two terms are automatically generated in order
                    # to make the parameter symmetric.
                    # In other words, specifying only this parameter:
                    # PARAMETER G(FCC_A1,AL,CR,NI;0) 298.15  +30300; 6000 N !
                    # Actually implies:
                    # PARAMETER G(FCC_A1,AL,CR,NI;0) 298.15  +30300; 6000 N !
                    # PARAMETER G(FCC_A1,AL,CR,NI;1) 298.15  +30300; 6000 N !
                    # PARAMETER G(FCC_A1,AL,CR,NI;2) 298.15  +30300; 6000 N !
                    #
                    # If either 1 or 2 is specified, no implicit parameters are
                    # generated.
                    # We need to handle this case.
                    if param['parameter_order'] == 0:
                        # are _any_ of the other parameter_orders specified?
                        ternary_param_query = (
                            (where('phase_name') == param['phase_name']) & \
                            (where('parameter_type') == \
                                param['parameter_type']) & \
                            (where('constituent_array') == \
                                param['constituent_array'])
                        )
                        other_tern_params = param_search(ternary_param_query)
                        if len(other_tern_params) == 1 and \
                            other_tern_params[0] == param:
                            # only the current parameter is specified
                            # We need to generate the other two parameters.
                            order_one = copy.deepcopy(param)
                            order_one['parameter_order'] = 1
                            order_two = copy.deepcopy(param)
                            order_two['parameter_order'] = 2
                            # Add these parameters to our iteration.
                            params.extend((order_one, order_two))
                    # Include variable indicated by parameter order index
                    # Perform Muggianu adjustment to site fractions
                    mixing_term *= comp_symbols[param['parameter_order']].subs(
                        self._Muggianu_correction_dict(comp_symbols),
                        simultaneous=True)
            mixing_term = mixing_term.xreplace(variable_rename_dict)
            mt_expand = mixing_term.expand()
            if not isinstance(mt_expand, Add):
                mt_expand = [mt_expand]
            else:
                mt_expand = mt_expand.args

            for arg in mt_expand:
                dof_param = np.zeros(len(dof)+1)
                dof_param[len(dof)] = 1
                if not isinstance(arg, Mul):
                    mulargs = [arg]
                else:
                    mulargs = arg.args
                for mularg in mulargs:
                    if isinstance(mularg, Pow):
                        if dof.index(mularg.base) == -1:
                            raise ValueError('Missing variable from degrees of freedom: ', mularg.base)
                        dof_param[dof.index(mularg.base)] = mularg.exp
                    elif isinstance(mularg, (Symbol)):
                        if dof.index(mularg) == -1:
                            raise ValueError('Missing variable from degrees of freedom: ', mularg)
                        dof_param[dof.index(mularg)] = 1
                    elif isinstance(mularg, (Float, Integer, Rational)):
                        dof_param[len(dof)] *= float(mularg)
                    else:
                        raise NotImplementedError(type(mularg), mularg)
                filled_param = self.symbol_replace(param['parameter'], all_symbols)
                build_piecewise_matrix(filled_param, [0,0,0,0] + dof_param.tolist(), 0, 10000,
                                       self.output_matrix, self.symbol_matrix, param_symbols)
            rk_terms.append(mixing_term * param['parameter'])
        self.output_matrix = np.array(self.output_matrix)
        result = Add(*rk_terms)

    def _eval(self, pressure, temperature, dof):
        result = 0.0
        eval_row = np.zeros(4+len(dof))
        eval_row[0] = pressure
        eval_row[1] = temperature
        eval_row[2] = np.log(pressure)
        eval_row[3] = np.log(temperature)
        eval_row[4:] = dof
        for row in self.output_matrix:
            if (temperature >= row[0]) and (temperature < row[1]):
                result += np.prod(np.power(eval_row, row[2:2+eval_row.shape[0]])) * row[2+eval_row.shape[0]] * row[2+eval_row.shape[0]+1]
        return result

    @staticmethod
    def _Muggianu_correction_dict(comps): #pylint: disable=C0103
        """
        Replace y_i -> y_i + (1 - sum(y involved in parameter)) / m,
        where m is the arity of the interaction parameter.
        Returns a dict converting the list of Symbols (comps) to this.
        m is assumed equal to the length of comps.

        When incorporating binary, ternary or n-ary interaction parameters
        into systems with more than n components, the sum of site fractions
        involved in the interaction parameter may no longer be unity. This
        breaks the symmetry of the parameter. The solution suggested by
        Muggianu, 1975, is to renormalize the site fractions by replacing them
        with a term that will sum to unity even in higher-order systems.
        There are other solutions that involve retaining the asymmetry for
        physical reasons, but this solution works well for components that
        are physically similar.

        This procedure is based on an analysis by Hillert, 1980,
        published in the Calphad journal.
        """
        arity = len(comps)
        return_dict = {}
        correction_term = (S.One - Add(*comps)) / arity
        for comp in comps:
            return_dict[comp] = comp + correction_term
        return return_dict

    @staticmethod
    def symbol_replace(obj, symbols):
        """
        Substitute values of symbols into 'obj'.

        Parameters
        ----------
        obj : SymPy object
        symbols : dict mapping sympy.Symbol to SymPy object

        Returns
        -------
        SymPy object
        """
        try:
            # Need to do more substitutions to catch symbols that are functions
            # of other symbols
            for iteration in range(_MAX_PARAM_NESTING):
                obj = obj.xreplace(symbols)
                undefs = obj.atoms(Symbol) - obj.atoms(v.StateVariable)
                if len(undefs) == 0:
                    break
        except AttributeError:
            # Can't use xreplace on a float
            pass
        return obj

class CompiledModel(Model):
    def __init__(self, dbe, comps, phase_name, parameters=None):
        super(CompiledModel, self).__init__(dbe, comps, phase_name, parameters=parameters)
        parameters = parameters if parameters is not None else {}
        phase = dbe.phases[phase_name]
        self.sublattice_dof = np.array([len(c) for c in self.constituents])
        self.site_ratios = np.array(self.site_ratios)
        # In the future, this should be bigger than num_sites.shape[0] to allow for multiple species
        # of the same type in the same sublattice for, e.g., same species with different charges
        self.composition_matrices = np.full((len(comps), self.site_ratios.shape[0], 2), -1.)
        if 'VA' in comps:
            self.vacancy_index = comps.index('VA')
        else:
            self.vacancy_index = -1
        var_idx = 0
        for variable in self.variables:
            if not isinstance(variable, v.SiteFraction):
                continue
            subl_index = variable.sublattice_index
            species = variable.species
            comp_index = comps.index(species)
            self.composition_matrices[comp_index, subl_index, 0] = self.site_ratios[subl_index]
            self.composition_matrices[comp_index, subl_index, 1] = var_idx
            var_idx += 1
        pure_param_query = (
            (where('phase_name') == phase_name) & \
            (where('parameter_order') == 0) & \
            (where('parameter_type') == "G") & \
            (where('constituent_array').test(self._purity_test))
        )
        excess_param_query = (
            (where('phase_name') == phase_name) & \
            ((where('parameter_type') == 'G') |
             (where('parameter_type') == 'L')) & \
            (where('constituent_array').test(self._interaction_test))
        )
        bm_param_query = (
            (where('phase_name') == phase_name) & \
            (where('parameter_type') == 'BMAGN') & \
            (where('constituent_array').test(self._array_validity))
        )
        tc_param_query = (
            (where('phase_name') == phase_name) & \
            (where('parameter_type') == 'TC') & \
            (where('constituent_array').test(self._array_validity))
        )
        all_symbols = dbe.symbols.copy()
        # Convert string symbol names to sympy Symbol objects
        # This makes xreplace work with the symbols dict
        all_symbols = dict([(Symbol(s), val) for s, val in all_symbols.items()])
        for param in parameters.keys():
            all_symbols.pop(param, None)
        pure_rksum = RedlichKisterSum(comps, dbe.phases[phase_name], dbe.search, pure_param_query, list(parameters.keys()), all_symbols)
        excess_rksum = RedlichKisterSum(comps, dbe.phases[phase_name], dbe.search, excess_param_query, list(parameters.keys()), all_symbols)
        tc_rksum = RedlichKisterSum(comps, dbe.phases[phase_name], dbe.search, tc_param_query, list(parameters.keys()), all_symbols)
        bm_rksum = RedlichKisterSum(comps, dbe.phases[phase_name], dbe.search, bm_param_query, list(parameters.keys()), all_symbols)
        self.pure_coef_matrix = pure_rksum.output_matrix
        self.pure_coef_symbol_matrix = pure_rksum.symbol_matrix
        self.excess_coef_matrix = excess_rksum.output_matrix
        self.excess_coef_symbol_matrix = excess_rksum.symbol_matrix
        self.bm_coef_matrix = bm_rksum.output_matrix
        self.bm_coef_symbol_matrix = bm_rksum.symbol_matrix
        self.tc_coef_matrix = tc_rksum.output_matrix
        self.tc_coef_symbol_matrix = tc_rksum.symbol_matrix
        self.ihj_magnetic_structure_factor = dbe.phases[phase_name].model_hints.get('ihj_magnetic_structure_factor', -1)
        self.afm_factor = dbe.phases[phase_name].model_hints.get('ihj_magnetic_afm_factor', 0)
        ordered_phase_name = phase.model_hints.get('ordered_phase', None)
        disordered_phase_name = phase.model_hints.get('disordered_phase', None)
        if (ordered_phase_name == phase_name) and (ordered_phase_name != disordered_phase_name):
            disordered_model = self.__class__(dbe, comps, disordered_phase_name, parameters=parameters)
            self.ordered = True
            self.disordered_sublattice_dof = disordered_model.sublattice_dof
            self.disordered_site_ratios = disordered_model.site_ratios
            # In the future, this should be bigger than num_sites.shape[0] to allow for multiple species
            # of the same type in the same sublattice for, e.g., same species with different charges
            self.disordered_composition_matrices = np.full((len(comps), self.disordered_site_ratios.shape[0], 2), -1.)
            var_idx = 0
            for variable in disordered_model.variables:
                if not isinstance(variable, v.SiteFraction):
                    continue
                subl_index = variable.sublattice_index
                species = variable.species
                comp_index = comps.index(species)
                self.disordered_composition_matrices[comp_index, subl_index, 0] = self.disordered_site_ratios[subl_index]
                self.disordered_composition_matrices[comp_index, subl_index, 1] = var_idx
                var_idx += 1
            self.disordered_pure_coef_matrix = disordered_model.pure_coef_matrix
            self.disordered_pure_coef_symbol_matrix = disordered_model.pure_coef_symbol_matrix
            self.disordered_excess_coef_matrix = disordered_model.excess_coef_matrix
            self.disordered_excess_coef_symbol_matrix = disordered_model.excess_coef_symbol_matrix
            self.disordered_bm_coef_matrix = disordered_model.bm_coef_matrix
            self.disordered_bm_coef_symbol_matrix = disordered_model.bm_coef_symbol_matrix
            self.disordered_tc_coef_matrix = disordered_model.tc_coef_matrix
            self.disordered_tc_coef_symbol_matrix = disordered_model.tc_coef_symbol_matrix
            self.disordered_ihj_magnetic_structure_factor = disordered_model.ihj_magnetic_structure_factor
            self.disordered_afm_factor = disordered_model.afm_factor
        else:
            self.ordered = False
    def get_obj_ptr(self):
        pass
    def get_grad_ptr(self):
        pass
    def get_hess_ptr(self):
        pass

cdef _eval_energy(cmpmdl, out, dof, parameters, out_idx, sign):
    cdef np.ndarray[ndim=1, dtype=np.float64_t] eval_row = np.zeros(2+dof.shape[0])
    cdef np.ndarray[ndim=1, dtype=np.float64_t] disordered_eval_row
    cdef np.ndarray[ndim=1, dtype=np.float64_t] disordered_dof
    cdef np.ndarray[ndim=1, dtype=np.float64_t]  row
    cdef double mass_normalization_factor = 0
    cdef double disordered_mass_normalization_factor = 0
    cdef double curie_temp = 0
    cdef double disordered_curie_temp = 0
    cdef double tau = 0
    cdef double bmagn = 0
    cdef double disordered_bmagn = 0
    cdef double A, p, res_tau
    cdef double out_energy = 0
    cdef int prev_idx = 0
    cdef int dof_idx
    eval_row[0] = dof[0]
    eval_row[1] = dof[1]
    eval_row[2] = log(dof[0])
    eval_row[3] = log(dof[1])
    eval_row[4:] = dof[2:]

    # Ideal mixing
    for entry_idx in range(cmpmdl.site_ratios.shape[0]):
        for dof_idx in range(prev_idx, prev_idx+cmpmdl.sublattice_dof[entry_idx]):
            if dof[2+dof_idx] > 1e-16:
                out_energy += 8.3145 * dof[1] * cmpmdl.site_ratios[entry_idx] * dof[2+dof_idx] * log(dof[2+dof_idx])
        prev_idx += cmpmdl.sublattice_dof[entry_idx]

    # End-member contribution
    for row in cmpmdl.pure_coef_matrix:
        if (dof[1] >= row[0]) and (dof[1] < row[1]):
            out_energy+= np.prod(np.power(eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * row[row.shape[0]-1]
    for row in cmpmdl.pure_coef_symbol_matrix:
        if (dof[1] >= row[0]) and (dof[1] < row[1]):
            out_energy += np.prod(np.power(eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * parameters[<int>row[row.shape[0]-1]]
    print(out_energy)
    # Interaction contribution
    for row in cmpmdl.excess_coef_matrix:
        if (dof[1] >= row[0]) and (dof[1] < row[1]):
            out_energy += np.prod(np.power(eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * row[row.shape[0]-1]
    for row in cmpmdl.excess_coef_symbol_matrix:
        if (dof[1] >= row[0]) and (dof[1] < row[1]):
            out_energy += np.prod(np.power(eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * parameters[<int>row[row.shape[0]-1]]
    # Magnetic contribution
    for row in cmpmdl.tc_coef_matrix:
        if (dof[1] >= row[0]) and (dof[1] < row[1]):
            curie_temp += np.prod(np.power(eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * row[row.shape[0]-1]
    for row in cmpmdl.tc_coef_symbol_matrix:
        if (dof[1] >= row[0]) and (dof[1] < row[1]):
            curie_temp += np.prod(np.power(eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * parameters[<int>row[row.shape[0]-1]]
    for row in cmpmdl.bm_coef_matrix:
        if (dof[1] >= row[0]) and (dof[1] < row[1]):
            bmagn += np.prod(np.power(eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * row[row.shape[0]-1]
    for row in cmpmdl.bm_coef_symbol_matrix:
        if (dof[1] >= row[0]) and (dof[1] < row[1]):
            bmagn += np.prod(np.power(eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * parameters[<int>row[row.shape[0]-1]]
    print(out_energy)
    if (curie_temp != 0) and (bmagn != 0) and (cmpmdl.ihj_magnetic_structure_factor > 0) and (cmpmdl.afm_factor != 0):
        if bmagn < 0:
            bmagn /= cmpmdl.afm_factor
        if curie_temp < 0:
            curie_temp /= cmpmdl.afm_factor
        if curie_temp > 1e-6:
            tau = dof[1] / curie_temp
            # define model parameters
            p = cmpmdl.ihj_magnetic_structure_factor
            A = 518./1125 + (11692./15975)*(1./p - 1.)
            # factor when tau < 1
            if tau < 1:
                res_tau = 1 - (1./A) * ((79./(140*p))*(tau**(-1)) + (474./497)*(1./p - 1) \
                    * ((tau**3)/6 + (tau**9)/135 + (tau**15)/600)
                                  )
            else:
                # factor when tau >= 1
                res_tau = -(1/A) * ((tau**-5)/10 + (tau**-15)/315. + (tau**-25)/1500.)
            out_energy += 8.3145 * dof[1] * log(bmagn+1) * res_tau
    for subl_idx in range(cmpmdl.site_ratios.shape[0]):
        if (cmpmdl.vacancy_index > -1) and cmpmdl.composition_matrices[cmpmdl.vacancy_index, subl_idx, 1] > -1:
            mass_normalization_factor += cmpmdl.site_ratios[subl_idx] * (1-dof[2+<int>cmpmdl.composition_matrices[cmpmdl.vacancy_index, subl_idx, 1]])
        else:
            mass_normalization_factor += cmpmdl.site_ratios[subl_idx]
    out_energy /= mass_normalization_factor
    print(out_energy)
    out[out_idx] = out[out_idx] + sign * out_energy

cpdef eval_energy(cmpmdl, out, dof, parameters, bounds):
    cdef np.ndarray[ndim=1, dtype=np.float64_t] eval_row = np.zeros(2+dof.shape[0])
    cdef np.ndarray[ndim=1, dtype=np.float64_t] disordered_eval_row
    cdef np.ndarray[ndim=1, dtype=np.float64_t] disordered_dof, ordered_dof
    cdef np.ndarray[ndim=1, dtype=np.float64_t]  row
    cdef double mass_normalization_factor = 0
    cdef double disordered_mass_normalization_factor = 0
    cdef double curie_temp = 0
    cdef double disordered_curie_temp = 0
    cdef double tau = 0
    cdef double bmagn = 0
    cdef double disordered_bmagn = 0
    cdef double A, p, res_tau
    cdef double disordered_energy = 0
    cdef int prev_idx = 0
    cdef int dof_idx
    out[0] = 0
    eval_row[0] = dof[0]
    eval_row[1] = dof[1]
    eval_row[2] = log(dof[0])
    eval_row[3] = log(dof[1])
    eval_row[4:] = dof[2:]

    _eval_energy(cmpmdl, out, dof, parameters, 0, 1)

    if cmpmdl.ordered is True:
        # Disordered phase contribution
        # Assume: Same components in all sublattices, except maybe a pure VA sublattice at the end
        disordered_dof = np.zeros(np.sum(cmpmdl.disordered_sublattice_dof)+2)
        disordered_dof[0] = dof[0]
        disordered_dof[1] = dof[1]
        ordered_dof = np.zeros(dof.shape[0])
        ordered_dof[0] = dof[0]
        ordered_dof[1] = dof[1]
        disordered_eval_row = np.zeros(disordered_dof.shape[0]+2)
        disordered_eval_row[0] = dof[0]
        disordered_eval_row[1] = dof[1]
        num_comps = cmpmdl.sublattice_dof[0]
        # Last sublattice is different from first; probably an interstitial sublattice
        # It should be treated separately
        if cmpmdl.sublattice_dof[0] != cmpmdl.sublattice_dof[cmpmdl.sublattice_dof.shape[0]-1]:
            site_sum = float(np.sum(cmpmdl.site_ratios[:cmpmdl.site_ratios.shape[0]-1]))
            for subl_idx in range(cmpmdl.site_ratios.shape[0]-1):
                for comp_idx in range(cmpmdl.sublattice_dof[subl_idx]):
                    disordered_dof[comp_idx+2] += (cmpmdl.site_ratios[subl_idx] / site_sum) * dof[subl_idx * num_comps + comp_idx + 2]
            dof_idx = np.sum(cmpmdl.sublattice_dof[:cmpmdl.sublattice_dof.shape[0]-1])
            disordered_dof_idx = np.sum(cmpmdl.disordered_sublattice_dof[:cmpmdl.disordered_sublattice_dof.shape[0]-1])
            # Copy interstitial values directly
            disordered_dof[disordered_dof_idx+2:] = dof[dof_idx+2:]
        else:
            site_sum = float(np.sum(cmpmdl.site_ratios))
            for subl_idx in range(cmpmdl.site_ratios.shape[0]):
                for comp_idx in range(cmpmdl.sublattice_dof[subl_idx]):
                    disordered_dof[subl_idx+2] += (cmpmdl.site_ratios[subl_idx] / site_sum) * dof[subl_idx * num_comps + comp_idx + 2]
        disordered_eval_row[0] = disordered_dof[0]
        disordered_eval_row[1] = disordered_dof[1]
        disordered_eval_row[2] = log(disordered_dof[0])
        disordered_eval_row[3] = log(disordered_dof[1])
        disordered_eval_row[4:] = disordered_dof[2:]
        # Ideal mixing
        prev_idx = 0
        for entry_idx in range(cmpmdl.disordered_site_ratios.shape[0]):
            for dof_idx in range(prev_idx, prev_idx+cmpmdl.disordered_sublattice_dof[entry_idx]):
                if disordered_dof[2+dof_idx] > 1e-16:
                    disordered_energy += 8.3145 * disordered_dof[1] * cmpmdl.disordered_site_ratios[entry_idx] * disordered_dof[2+dof_idx] * log(disordered_dof[2+dof_idx])
            prev_idx += cmpmdl.disordered_sublattice_dof[entry_idx]
        # End-member contribution
        for row in cmpmdl.disordered_pure_coef_matrix:
            if (disordered_dof[1] >= row[0]) and (disordered_dof[1] < row[1]):
                disordered_energy += np.prod(np.power(disordered_eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * row[row.shape[0]-1]
        for row in cmpmdl.disordered_pure_coef_symbol_matrix:
            if (disordered_dof[1] >= row[0]) and (disordered_dof[1] < row[1]):
                disordered_energy += np.prod(np.power(disordered_eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * parameters[<int>row[row.shape[0]-1]]
        # Interaction contribution
        for row in cmpmdl.disordered_excess_coef_matrix:
            if (disordered_dof[1] >= row[0]) and (disordered_dof[1] < row[1]):
                disordered_energy += np.prod(np.power(disordered_eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * row[row.shape[0]-1]
        for row in cmpmdl.disordered_excess_coef_symbol_matrix:
            if (disordered_dof[1] >= row[0]) and (disordered_dof[1] < row[1]):
                disordered_energy += np.prod(np.power(disordered_eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * parameters[<int>row[row.shape[0]-1]]
        # Magnetic contribution
        for row in cmpmdl.disordered_tc_coef_matrix:
            if (disordered_dof[1] >= row[0]) and (disordered_dof[1] < row[1]):
                disordered_curie_temp += np.prod(np.power(disordered_eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * row[row.shape[0]-1]
        for row in cmpmdl.disordered_tc_coef_symbol_matrix:
            if (disordered_dof[1] >= row[0]) and (disordered_dof[1] < row[1]):
                disordered_curie_temp += np.prod(np.power(disordered_eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * parameters[<int>row[row.shape[0]-1]]
        for row in cmpmdl.disordered_bm_coef_matrix:
            if (disordered_dof[1] >= row[0]) and (disordered_dof[1] < row[1]):
                disordered_bmagn += np.prod(np.power(disordered_eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * row[row.shape[0]-1]
        for row in cmpmdl.disordered_bm_coef_symbol_matrix:
            if (disordered_dof[1] >= row[0]) and (disordered_dof[1] < row[1]):
                disordered_bmagn += np.prod(np.power(disordered_eval_row, row[2:row.shape[0]-2])) * row[row.shape[0]-2] * parameters[<int>row[row.shape[0]-1]]
        if (disordered_curie_temp != 0) and (disordered_bmagn != 0) and (cmpmdl.disordered_ihj_magnetic_structure_factor > 0) and (cmpmdl.disordered_afm_factor != 0):
            if disordered_bmagn < 0:
                disordered_bmagn /= cmpmdl.disordered_afm_factor
            if disordered_curie_temp < 0:
                disordered_curie_temp /= cmpmdl.disordered_afm_factor
            if disordered_curie_temp > 1e-6:
                tau = dof[1] / curie_temp
                # define model parameters
                p = cmpmdl.disordered_ihj_magnetic_structure_factor
                A = 518./1125 + (11692./15975)*(1./p - 1.)
                # factor when tau < 1
                if tau < 1:
                    res_tau = 1 - (1./A) * ((79./(140*p))*(tau**(-1)) + (474./497)*(1./p - 1) \
                        * ((tau**3)/6 + (tau**9)/135 + (tau**15)/600)
                                      )
                else:
                    # factor when tau >= 1
                    res_tau = -(1/A) * ((tau**-5)/10 + (tau**-15)/315. + (tau**-25)/1500.)
                disordered_energy += 8.3145 * disordered_dof[1] * log(disordered_bmagn+1) * res_tau
        for subl_idx in range(cmpmdl.disordered_site_ratios.shape[0]):
            if (cmpmdl.vacancy_index > -1) and cmpmdl.disordered_composition_matrices[cmpmdl.vacancy_index, subl_idx, 1] > -1:
                disordered_mass_normalization_factor += cmpmdl.disordered_site_ratios[subl_idx] * (1-disordered_dof[2+<int>cmpmdl.disordered_composition_matrices[cmpmdl.vacancy_index, subl_idx, 1]])
            else:
                disordered_mass_normalization_factor += cmpmdl.disordered_site_ratios[subl_idx]
        disordered_energy /= disordered_mass_normalization_factor
        # Subtract ordered energy at disordered configuration
        ordered_dof[0] = dof[0]
        ordered_dof[1] = dof[1]
        if cmpmdl.sublattice_dof[0] != cmpmdl.sublattice_dof[cmpmdl.sublattice_dof.shape[0]-1]:
            for subl_idx in range(cmpmdl.site_ratios.shape[0]-1):
                for comp_idx in range(cmpmdl.sublattice_dof[subl_idx]):
                    ordered_dof[subl_idx * num_comps + comp_idx + 2] = disordered_dof[comp_idx+2]
            dof_idx = np.sum(cmpmdl.sublattice_dof[:cmpmdl.sublattice_dof.shape[0]-1])
            disordered_dof_idx = np.sum(cmpmdl.disordered_sublattice_dof[:cmpmdl.disordered_sublattice_dof.shape[0]-1])
            # Copy interstitial values directly
            ordered_dof[dof_idx+2:] = disordered_dof[disordered_dof_idx+2:]
        else:
            for subl_idx in range(cmpmdl.site_ratios.shape[0]):
                for comp_idx in range(cmpmdl.sublattice_dof[subl_idx]):
                    ordered_dof[subl_idx * num_comps + comp_idx + 2] = disordered_dof[subl_idx+2]
        print('ordered_dof', ordered_dof)
        print('main_ord', out[0])
        main_ord = out[0]
        _eval_energy(cmpmdl, out, ordered_dof, parameters, 0, -1)
        print('main-sub',out[0]-main_ord)
        print('disordered_energy', disordered_energy)
        out[0] += disordered_energy

def eval_gradient_energy(cmpmdl, pressure, temperature, dof, out):
    pass
def eval_hessian_energy(cmpmdl, pressure, temperature, dof, out):
    pass