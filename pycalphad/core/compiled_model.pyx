# cython: profile=True
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log
from sympy import Symbol
from tinydb import where
from collections import OrderedDict
from pycalphad.core.rksum import RedlichKisterSum
from pycalphad.core.sympydiff_utils import build_functions
import pycalphad.variables as v
from pycalphad import Model
from pycalphad.model import DofError
from cpython cimport PY_VERSION_HEX, PyCObject_Check, PyCObject_AsVoidPtr, PyCapsule_CheckExact, PyCapsule_GetPointer

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _sum(double[:] arr) nogil:
    cdef double result = 0
    cdef int idx
    for idx in range(arr.shape[0]):
        result += arr[idx]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _intsum(int[:] arr) nogil:
    cdef int result = 0
    cdef int idx
    for idx in range(arr.shape[0]):
        result += arr[idx]
    return result

# From https://gist.github.com/pv/5437087
cdef void* f2py_pointer(obj):
    if PY_VERSION_HEX < 0x03000000:
        if (PyCObject_Check(obj)):
            return PyCObject_AsVoidPtr(obj)
    elif PY_VERSION_HEX >= 0x02070000:
        if (PyCapsule_CheckExact(obj)):
            return PyCapsule_GetPointer(obj, NULL);
    raise ValueError("Not an object containing a void ptr")

# Forward declaration necessary for some self-referencing below
cdef public class CompiledModel(object)[type CompiledModelType, object CompiledModelObject]

cdef public class CompiledModel(object)[type CompiledModelType, object CompiledModelObject]:
    def __cinit__(self, dbe, comps, phase_name, parameters=None, _debug=False):
        possible_comps = set([x.upper() for x in comps])
        comps = sorted(comps, key=str)
        phase = dbe.phases[phase_name]
        self.constituents = []
        self.components = set()
        # Verify that this phase is still possible to build
        for sublattice in phase.constituents:
            if len(set(sublattice).intersection(possible_comps)) == 0:
                # None of the components in a sublattice are active
                # We cannot build a model of this phase
                raise DofError(
                    '{0}: Sublattice {1} of {2} has no components in {3}' \
                    .format(phase_name, sublattice,
                            phase.constituents,
                            self.components))
            self.components |= set(sublattice).intersection(possible_comps)
            self.constituents.append(sorted(set(sublattice).intersection(self.components)))
        self.components = sorted(self.components, key=str)
        self.variables = []
        for idx, sublattice in enumerate(self.constituents):
            for comp in sublattice:
                self.variables.append(v.Y(phase_name, idx, comp))
        parameters = parameters if parameters is not None else {}
        self._debug = _debug
        if _debug:
            debugmodel = Model(dbe, comps, phase_name, parameters)
            out = debugmodel.energy
            if isinstance(parameters, dict):
                parameters = OrderedDict(sorted(parameters.items(), key=str))
            param_symbols = tuple(parameters.keys())
            undefs = list(out.atoms(Symbol) - out.atoms(v.StateVariable) - set(param_symbols))
            for undef in undefs:
                out = out.xreplace({undef: float(0)})
            _debugobj, _debuggrad, _debughess = build_functions(out, tuple([v.P, v.T] + self.variables),
                                                                parameters=param_symbols)
            # Trigger lazy computation
            if _debugobj is not None:
                _debugobj.kernel
                self._debugobj = <func_t*> f2py_pointer(_debugobj._kernel._cpointer)
            if _debuggrad is not None:
                _debuggrad.kernel
                self._debuggrad = <func_novec_t*> f2py_pointer(_debuggrad._kernel._cpointer)
            if _debughess is not None:
                _debughess.kernel
                self._debughess = <func_novec_t*> f2py_pointer(_debughess._kernel._cpointer)

        self.site_ratios = np.array(phase.sublattices)
        self.sublattice_dof = np.array([len(c) for c in self.constituents], dtype=np.int32)
        # In the future, this should be bigger than num_sites.shape[0] to allow for multiple species
        # of the same type in the same sublattice for, e.g., same species with different charges
        self.composition_matrices = np.full((len(comps), len(self.site_ratios), 2), -1.)
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
            disordered_model = CompiledModel(dbe, comps, disordered_phase_name, parameters=parameters)
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

    def _purity_test(self, constituent_array):
        """
        Check if constituent array only has one species in its array
        This species must also be an active species
        """
        for sublattice in constituent_array:
            if len(sublattice) != 1:
                return False
            if (sublattice[0] not in self.components) and \
                (sublattice[0] != '*'):
                return False
        return True

    def _array_validity(self, constituent_array):
        """
        Check that the current array contains only active species.
        """
        for sublattice in constituent_array:
            valid = set(sublattice).issubset(self.components) \
                or sublattice[0] == '*'
            if not valid:
                return False
        return True

    def _interaction_test(self, constituent_array):
        """
        Check if constituent array has more than one active species in
        its array for at least one sublattice.
        """
        result = False
        for sublattice in constituent_array:
            # check if all elements involved are also active
            valid = set(sublattice).issubset(self.components) \
                or sublattice[0] == '*'
            if len(sublattice) > 1 and valid:
                result = True
            if not valid:
                result = False
                break
        return result

    @cython.boundscheck(False)
    cdef double _eval_rk_matrix(self, double[:,:] coef_mat, double[:,:] symbol_mat,
                                double[:] eval_row, double[:] parameters) nogil:
        cdef double result = 0
        cdef double prod_result
        cdef int row_idx1 = 0
        cdef int row_idx2 = 0
        cdef int col_idx = 0
        if coef_mat.shape[1] > 0:
            for row_idx1 in range(coef_mat.shape[0]):
                if (eval_row[1] >= coef_mat[row_idx1, 0]) and (eval_row[1] < coef_mat[row_idx1, 1]):
                    prod_result = coef_mat[row_idx1, coef_mat.shape[1]-2] * coef_mat[row_idx1, coef_mat.shape[1]-1]
                    for col_idx in range(coef_mat.shape[1]-4):
                        prod_result = prod_result * (eval_row[col_idx] ** coef_mat[row_idx1, 2+col_idx])
                    result += prod_result
        if symbol_mat.shape[1] > 0:
            for row_idx2 in range(symbol_mat.shape[0]):
                if (eval_row[1] >= symbol_mat[row_idx2, 0]) and (eval_row[1] < symbol_mat[row_idx2, 1]):
                    prod_result = symbol_mat[row_idx2, symbol_mat.shape[1]-2] * parameters[<int>symbol_mat[row_idx2, symbol_mat.shape[1]-1]]
                    for col_idx in range(symbol_mat.shape[1]-4):
                        prod_result = prod_result * (eval_row[col_idx] ** symbol_mat[row_idx2, 2+col_idx])
                    result += prod_result
        return result

    cdef void _eval_rk_matrix_gradient(self, double[:] out, double[:,:] coef_mat, double[:,:] symbol_mat,
                                       double[:] eval_row, double[:] parameters):
        cdef double result = 0
        cdef double prod_result
        cdef int row_idx1 = 0
        cdef int row_idx2 = 0
        cdef int col_idx = 0
        cdef int dof_idx
        # eval_row order: P,T,ln(P),ln(T),y...
        # dof order: P,T,y...
        # coef_mat order: low_temp,high_temp,P,T,ln(P),ln(T),y...,constant_term,parameter_value
        for dof_idx in range(eval_row.shape[0]-2):
            if coef_mat.shape[1] > 0:
                for row_idx1 in range(coef_mat.shape[0]):
                    if (eval_row[1] >= coef_mat[row_idx1, 0]) and (eval_row[1] < coef_mat[row_idx1, 1]):
                        if dof_idx < 2:
                            # special handling for state variables since they also can have a ln term
                            if (coef_mat[row_idx1, 2+dof_idx] != 0) or (coef_mat[row_idx1, 4+dof_idx] != 0):
                                prod_result = coef_mat[row_idx1, coef_mat.shape[1]-2] * coef_mat[row_idx1, coef_mat.shape[1]-1]
                                prod_result *= eval_row[dof_idx] ** (coef_mat[row_idx1, 2+dof_idx] - 1)
                                prod_result *= eval_row[2+dof_idx] ** (coef_mat[row_idx1, 4+dof_idx] - 1)
                                prod_result *= coef_mat[row_idx1, 2+dof_idx] * eval_row[2+dof_idx] + coef_mat[row_idx1, 4+dof_idx]
                                for col_idx in range(4, coef_mat.shape[1]-4):
                                        prod_result *= (eval_row[col_idx] ** coef_mat[row_idx1, 2+col_idx])
                                out[dof_idx] += prod_result
                        else:
                            if coef_mat[row_idx1, 4+dof_idx] != 0:
                                prod_result = coef_mat[row_idx1, coef_mat.shape[1]-2] * coef_mat[row_idx1, coef_mat.shape[1]-1]
                                for col_idx in range(coef_mat.shape[1]-4):
                                    if col_idx == 2+dof_idx:
                                        prod_result *= (coef_mat[row_idx1, 2+col_idx] * eval_row[col_idx] ** (coef_mat[row_idx1, 2+col_idx] - 1))
                                    else:
                                        prod_result *= (eval_row[col_idx] ** coef_mat[row_idx1, 2+col_idx])
                                out[dof_idx] += prod_result
            if symbol_mat.shape[1] > 0:
                for row_idx2 in range(symbol_mat.shape[0]):
                    if (eval_row[1] >= symbol_mat[row_idx2, 0]) and (eval_row[1] < symbol_mat[row_idx2, 1]):
                        if dof_idx < 2:
                            # special handling for state variables since they also can have a ln term
                            if (coef_mat[row_idx2, dof_idx] != 0) or (coef_mat[row_idx2, 2+dof_idx] != 0):
                                prod_result = coef_mat[row_idx2, coef_mat.shape[1]-2] * parameters[<int>symbol_mat[row_idx2, symbol_mat.shape[1]-1]]
                                prod_result *= eval_row[dof_idx] ** (coef_mat[row_idx2, 2+dof_idx] - 1)
                                prod_result *= eval_row[2+dof_idx] ** (coef_mat[row_idx2, 4+dof_idx] - 1)
                                prod_result *= coef_mat[row_idx2, 2+dof_idx] * eval_row[2+dof_idx] + coef_mat[row_idx2, 4+dof_idx]
                                for col_idx in range(4, coef_mat.shape[1]-4):
                                        prod_result *= (eval_row[col_idx] ** coef_mat[row_idx2, 2+col_idx])
                                out[dof_idx] += prod_result
                        else:
                            if coef_mat[row_idx1, 4+dof_idx] != 0:
                                prod_result = coef_mat[row_idx2, coef_mat.shape[1]-2] * parameters[<int>symbol_mat[row_idx2, symbol_mat.shape[1]-1]]
                                for col_idx in range(coef_mat.shape[1]-4):
                                    if col_idx == 2+dof_idx:
                                        prod_result *= (coef_mat[row_idx2, 2+col_idx] * eval_row[col_idx] ** (coef_mat[row_idx2, 2+col_idx] - 1))
                                    else:
                                        prod_result *= (eval_row[col_idx] ** coef_mat[row_idx2, 2+col_idx])
                                out[dof_idx] += prod_result

    @cython.boundscheck(False)
    cdef _eval_energy(self, double[::1] out, double[:,:] dof, double[:] parameters, double sign):
        cdef double[:,:] eval_row = np.zeros((dof.shape[0], 4+sum(self.sublattice_dof)))
        cdef double[:] mass_normalization_factor = np.zeros(dof.shape[0])
        cdef double[:] curie_temp = np.zeros(dof.shape[0])
        cdef double[:] tau = np.zeros(dof.shape[0])
        cdef double[:] bmagn = np.zeros(dof.shape[0])
        cdef double[:] res_tau = np.zeros(dof.shape[0])
        cdef double p = self.ihj_magnetic_structure_factor
        cdef double A = 518./1125 + (11692./15975)*(1./p - 1.)
        cdef double[:] out_energy = np.zeros(dof.shape[0])
        cdef int[:] prev_idx = np.zeros(dof.shape[0], dtype=np.int32)
        cdef int dof_idx, out_idx, eval_idx
        with nogil:
            for out_idx in range(out.shape[0]):
                eval_row[out_idx, 0] = dof[out_idx, 0]
                eval_row[out_idx, 1] = dof[out_idx, 1]
                eval_row[out_idx, 2] = log(dof[out_idx, 0])
                eval_row[out_idx, 3] = log(dof[out_idx, 1])
                for eval_idx in range(eval_row.shape[1]-4):
                    eval_row[out_idx, 4+eval_idx] = dof[out_idx, 2+eval_idx]
                # Ideal mixing
                prev_idx[out_idx] = 0
                for entry_idx in range(self.site_ratios.shape[0]):
                    for dof_idx in range(prev_idx[out_idx], prev_idx[out_idx]+self.sublattice_dof[entry_idx]):
                        if dof[out_idx, 2+dof_idx] > 1e-16:
                            out_energy[out_idx] += 8.3145 * dof[out_idx, 1] * self.site_ratios[entry_idx] * dof[out_idx, 2+dof_idx] * log(dof[out_idx, 2+dof_idx])
                    prev_idx[out_idx] += self.sublattice_dof[entry_idx]

                # End-member contribution
                out_energy[out_idx] += self._eval_rk_matrix(self.pure_coef_matrix, self.pure_coef_symbol_matrix,
                                                            eval_row[out_idx], parameters)
                # Interaction contribution
                out_energy[out_idx] += self._eval_rk_matrix(self.excess_coef_matrix, self.excess_coef_symbol_matrix,
                                                            eval_row[out_idx], parameters)
                # Magnetic contribution
                curie_temp[out_idx] = self._eval_rk_matrix(self.tc_coef_matrix, self.tc_coef_symbol_matrix,
                                                           eval_row[out_idx], parameters)
                bmagn[out_idx] = self._eval_rk_matrix(self.bm_coef_matrix, self.bm_coef_symbol_matrix,
                                                      eval_row[out_idx], parameters)
                if (curie_temp[out_idx] != 0) and (bmagn[out_idx] != 0) and (self.ihj_magnetic_structure_factor > 0) and (self.afm_factor != 0):
                    if bmagn[out_idx] < 0:
                        bmagn[out_idx] /= self.afm_factor
                    if curie_temp[out_idx] < 0:
                        curie_temp[out_idx] /= self.afm_factor
                    if curie_temp[out_idx] > 1e-6:
                        tau[out_idx] = dof[out_idx, 1] / curie_temp[out_idx]
                        # factor when tau < 1
                        if tau[out_idx] < 1:
                            res_tau[out_idx] = 1 - (1./A) * ((79./(140*p))*(tau[out_idx]**(-1)) + (474./497)*(1./p - 1) \
                                * ((tau[out_idx]**3)/6 + (tau[out_idx]**9)/135 + (tau[out_idx]**15)/600)
                                              )
                        else:
                            # factor when tau >= 1
                            res_tau[out_idx] = -(1/A) * ((tau[out_idx]**-5)/10 + (tau[out_idx]**-15)/315. + (tau[out_idx]**-25)/1500.)
                        out_energy[out_idx] += 8.3145 * dof[out_idx, 1] * log(bmagn[out_idx]+1) * res_tau[out_idx]
                for subl_idx in range(self.site_ratios.shape[0]):
                    if (self.vacancy_index > -1) and self.composition_matrices[self.vacancy_index, subl_idx, 1] > -1:
                        mass_normalization_factor[out_idx] += self.site_ratios[subl_idx] * (1-dof[out_idx, 2+<int>self.composition_matrices[self.vacancy_index, subl_idx, 1]])
                    else:
                        mass_normalization_factor[out_idx] += self.site_ratios[subl_idx]
                out_energy[out_idx] /= mass_normalization_factor[out_idx]
                out[out_idx] = out[out_idx] + sign * out_energy[out_idx]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _compute_disordered_dof(self, double[:,:] disordered_dof, double[:,:] dof) nogil:
        cdef int out_idx, dof_idx, comp_idx, subl_idx, disordered_dof_idx
        cdef int num_comps
        cdef double site_sum
        for out_idx in range(disordered_dof.shape[0]):
            site_sum = 0
            num_comps = 0
            # Disordered phase contribution
            # Assume: Same components in all sublattices, except maybe a pure VA sublattice at the end
            disordered_dof[out_idx, 0] = dof[out_idx, 0]
            disordered_dof[out_idx, 1] = dof[out_idx, 1]
            num_comps = self.sublattice_dof[0]
            # Last sublattice is different from first; probably an interstitial sublattice
            # It should be treated separately
            if self.sublattice_dof[0] != self.sublattice_dof[self.sublattice_dof.shape[0]-1]:
                site_sum = _sum(self.site_ratios[:self.site_ratios.shape[0]-1])
                for subl_idx in range(self.site_ratios.shape[0]-1):
                    for comp_idx in range(self.sublattice_dof[subl_idx]):
                        disordered_dof[out_idx, comp_idx+2] += (self.site_ratios[subl_idx] / site_sum) * dof[out_idx, subl_idx * num_comps + comp_idx + 2]
                dof_idx = _intsum(self.sublattice_dof[:self.sublattice_dof.shape[0]-1])
                disordered_dof_idx = _intsum(self.disordered_sublattice_dof[:self.disordered_sublattice_dof.shape[0]-1])
                # Copy interstitial values directly
                disordered_dof[out_idx, disordered_dof_idx+2:] = dof[out_idx, dof_idx+2:]
            else:
                site_sum = _sum(self.site_ratios)
                for subl_idx in range(self.site_ratios.shape[0]):
                    for comp_idx in range(self.sublattice_dof[subl_idx]):
                        disordered_dof[out_idx, subl_idx+2] += (self.site_ratios[subl_idx] / site_sum) * dof[out_idx, subl_idx * num_comps + comp_idx + 2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _compute_ordered_dof(self, double[:,:] ordered_dof, double[:,:] disordered_dof) nogil:
        cdef int dof_idx, out_idx, subl_idx, comp_idx, disordered_dof_idx
        cdef int num_comps = self.sublattice_dof[0]
        for out_idx in range(ordered_dof.shape[0]):
            # Subtract ordered energy at disordered configuration
            ordered_dof[out_idx, 0] = disordered_dof[out_idx, 0]
            ordered_dof[out_idx, 1] = disordered_dof[out_idx, 1]
            if self.sublattice_dof[0] != self.sublattice_dof[self.sublattice_dof.shape[0]-1]:
                for subl_idx in range(self.site_ratios.shape[0]-1):
                    for comp_idx in range(self.sublattice_dof[subl_idx]):
                        ordered_dof[out_idx, subl_idx * num_comps + comp_idx + 2] = disordered_dof[out_idx, comp_idx+2]
                dof_idx = _intsum(self.sublattice_dof[:self.sublattice_dof.shape[0]-1])
                disordered_dof_idx = _intsum(self.disordered_sublattice_dof[:self.disordered_sublattice_dof.shape[0]-1])
                # Copy interstitial values directly
                ordered_dof[out_idx, dof_idx+2:] = disordered_dof[out_idx, disordered_dof_idx+2:]
            else:
                for subl_idx in range(self.site_ratios.shape[0]):
                    for comp_idx in range(self.sublattice_dof[subl_idx]):
                        ordered_dof[out_idx, subl_idx * num_comps + comp_idx + 2] = disordered_dof[out_idx, subl_idx+2]

    cpdef eval_energy(self, double[::1] out, double[:,:] dof, double[:] parameters):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] eval_row = np.zeros(4+sum(self.sublattice_dof))
        cdef np.ndarray[ndim=1, dtype=np.float64_t] disordered_eval_row
        cdef np.ndarray[ndim=2, dtype=np.float64_t] disordered_dof
        cdef np.ndarray[ndim=2, dtype=np.float64_t] ordered_dof
        cdef np.ndarray[ndim=1, dtype=np.float64_t]  row
        cdef double disordered_mass_normalization_factor = 0
        cdef double disordered_curie_temp = 0
        cdef double tau = 0
        cdef double disordered_bmagn = 0
        cdef double A, p, res_tau
        cdef double disordered_energy = 0
        cdef int prev_idx = 0
        cdef int dof_idx, out_idx, subl_idx
        self._eval_energy(out, dof, parameters, 1)
        if self.ordered:
            disordered_dof = np.zeros((dof.shape[0], _intsum(self.disordered_sublattice_dof)+2))
            ordered_dof = np.zeros((dof.shape[0], dof.shape[1]))
            self._compute_disordered_dof(disordered_dof, dof)
            self._compute_ordered_dof(ordered_dof, disordered_dof)
            # Subtract ordered energy at disordered configuration
            self._eval_energy(out, ordered_dof, parameters, -1)
            disordered_eval_row = np.zeros(disordered_dof.shape[1]+2)
            for out_idx in range(out.shape[0]):
                disordered_mass_normalization_factor = 0
                disordered_curie_temp = 0
                disordered_bmagn = 0
                disordered_energy = 0
                eval_row[0] = dof[out_idx, 0]
                eval_row[1] = dof[out_idx, 1]
                eval_row[2] = log(dof[out_idx, 0])
                eval_row[3] = log(dof[out_idx, 1])
                eval_row[4:] = dof[out_idx, 2:]
                # Disordered phase contribution
                # Assume: Same components in all sublattices, except maybe a pure VA sublattice at the end
                disordered_eval_row[0] = disordered_dof[out_idx, 0]
                disordered_eval_row[1] = disordered_dof[out_idx, 1]
                disordered_eval_row[2] = log(disordered_dof[out_idx, 0])
                disordered_eval_row[3] = log(disordered_dof[out_idx, 1])
                disordered_eval_row[4:] = disordered_dof[out_idx, 2:]
                # Ideal mixing
                prev_idx = 0
                for entry_idx in range(self.disordered_site_ratios.shape[0]):
                    for dof_idx in range(prev_idx, prev_idx+self.disordered_sublattice_dof[entry_idx]):
                        if disordered_dof[out_idx, 2+dof_idx] > 1e-16:
                            disordered_energy += 8.3145 * disordered_dof[out_idx, 1] * self.disordered_site_ratios[entry_idx] * disordered_dof[out_idx, 2+dof_idx] * log(disordered_dof[out_idx, 2+dof_idx])
                    prev_idx += self.disordered_sublattice_dof[entry_idx]
                # End-member contribution
                disordered_energy += self._eval_rk_matrix(self.disordered_pure_coef_matrix,
                                                          self.disordered_pure_coef_symbol_matrix,
                                                          disordered_eval_row, parameters)
                # Interaction contribution
                disordered_energy += self._eval_rk_matrix(self.disordered_excess_coef_matrix,
                                                          self.disordered_excess_coef_symbol_matrix,
                                                          disordered_eval_row, parameters)
                # Magnetic contribution
                disordered_curie_temp += self._eval_rk_matrix(self.disordered_tc_coef_matrix,
                                                              self.disordered_tc_coef_symbol_matrix,
                                                              disordered_eval_row, parameters)
                disordered_bmagn += self._eval_rk_matrix(self.disordered_bm_coef_matrix,
                                                         self.disordered_bm_coef_symbol_matrix,
                                                         disordered_eval_row, parameters)
                if (disordered_curie_temp != 0) and (disordered_bmagn != 0) and (self.disordered_ihj_magnetic_structure_factor > 0) and (self.disordered_afm_factor != 0):
                    if disordered_bmagn < 0:
                        disordered_bmagn /= self.disordered_afm_factor
                    if disordered_curie_temp < 0:
                        disordered_curie_temp /= self.disordered_afm_factor
                    if disordered_curie_temp > 1e-6:
                        tau = dof[out_idx, 1] / disordered_curie_temp
                        # define model parameters
                        p = self.disordered_ihj_magnetic_structure_factor
                        A = 518./1125 + (11692./15975)*(1./p - 1.)
                        # factor when tau < 1
                        if tau < 1:
                            res_tau = 1 - (1./A) * ((79./(140*p))*(tau**(-1)) + (474./497)*(1./p - 1) \
                                * ((tau**3)/6 + (tau**9)/135 + (tau**15)/600)
                                              )
                        else:
                            # factor when tau >= 1
                            res_tau = -(1./A) * ((tau**-5)/10. + (tau**-15)/315. + (tau**-25)/1500.)
                        disordered_energy += 8.3145 * disordered_dof[out_idx, 1] * log(disordered_bmagn+1) * res_tau
                for subl_idx in range(self.disordered_site_ratios.shape[0]):
                    if (self.vacancy_index > -1) and self.disordered_composition_matrices[self.vacancy_index, subl_idx, 1] > -1:
                        disordered_mass_normalization_factor += self.disordered_site_ratios[subl_idx] * (1-disordered_dof[out_idx, 2+<int>self.disordered_composition_matrices[self.vacancy_index, subl_idx, 1]])
                    else:
                        disordered_mass_normalization_factor += self.disordered_site_ratios[subl_idx]
                disordered_energy /= disordered_mass_normalization_factor
                out[out_idx] += disordered_energy
        if self._debug:
            debugout = np.zeros_like(out)
            self._debug_energy(debugout, np.asfortranarray(dof), np.ascontiguousarray(parameters))
            np.testing.assert_allclose(out,debugout)

    cdef _debug_energy(self, double[::1] debugout, double[::1,:] dof, double[::1] parameters):
        if parameters.shape[0] == 0:
            self._debugobj(&debugout[0], &dof[0,0], NULL, <int*>&debugout.shape[0])
        else:
            self._debugobj(&debugout[0], &dof[0,0], &parameters[0], <int*>&debugout.shape[0])

    cdef _debug_energy_gradient(self, double[::1] debugout, double[::1] dof, double[::1] parameters):
        if parameters.shape[0] == 0:
            self._debuggrad(&dof[0], NULL, &debugout[0])
        else:
            self._debuggrad(&dof[0], &parameters[0], &debugout[0])

    cdef _eval_energy_gradient(self, double[::1] out_grad, double[:] dof, double[:] parameters, double sign):
        cdef double[:] eval_row = np.zeros(4+sum(self.sublattice_dof))
        cdef double[:] out = np.zeros(dof.shape[0])
        cdef double[::1] energy = np.atleast_1d(np.zeros(1))
        cdef double mass_normalization_factor = 0
        cdef double[:] mass_normalization_vacancy_factor = np.zeros(dof.shape[0]-2)
        cdef double curie_temp = 0
        cdef double[:] curie_temp_prime = np.zeros(dof.shape[0])
        cdef double tau = 0
        cdef double[:] tau_prime = np.zeros(dof.shape[0])
        cdef double bmagn = 0
        cdef double[:] bmagn_prime = np.zeros(dof.shape[0])
        cdef double g_func = 0
        cdef double g_func_prime = 0
        cdef double p = self.ihj_magnetic_structure_factor
        cdef double A = 518./1125 + (11692./15975)*(1./p - 1.)
        cdef int prev_idx = 0
        cdef int dof_idx, eval_idx
        eval_row[0] = dof[0]
        eval_row[1] = dof[1]
        eval_row[2] = log(dof[0])
        eval_row[3] = log(dof[1])
        for eval_idx in range(sum(self.sublattice_dof)):
            eval_row[4+eval_idx] = dof[2+eval_idx]
        # Ideal mixing
        for entry_idx in range(self.site_ratios.shape[0]):
            for dof_idx in range(prev_idx, prev_idx+self.sublattice_dof[entry_idx]):
                if dof[2+dof_idx] > 1e-16:
                    # wrt P: 0
                    # wrt T
                    out[1] += 8.3145 * self.site_ratios[entry_idx] * dof[2+dof_idx] * log(dof[2+dof_idx])
                # wrt y
                out[2+dof_idx] += 8.3145 * dof[1] * self.site_ratios[entry_idx] * (log(dof[2+dof_idx]) + 1)
            prev_idx += self.sublattice_dof[entry_idx]

        # End-member contribution
        self._eval_rk_matrix_gradient(out, self.pure_coef_matrix, self.pure_coef_symbol_matrix,
                                      eval_row, parameters)
        # Interaction contribution
        self._eval_rk_matrix_gradient(out, self.excess_coef_matrix, self.excess_coef_symbol_matrix,
                                      eval_row, parameters)
        # Magnetic contribution
        curie_temp = self._eval_rk_matrix(self.tc_coef_matrix, self.tc_coef_symbol_matrix,
                                          eval_row, parameters)
        bmagn = self._eval_rk_matrix(self.bm_coef_matrix, self.bm_coef_symbol_matrix,
                                     eval_row, parameters)
        if (curie_temp != 0) and (bmagn != 0) and (self.ihj_magnetic_structure_factor > 0) and (self.afm_factor != 0):
            self._eval_rk_matrix_gradient(curie_temp_prime, self.tc_coef_matrix, self.tc_coef_symbol_matrix,
                              eval_row, parameters)
            self._eval_rk_matrix_gradient(bmagn_prime, self.bm_coef_matrix, self.bm_coef_symbol_matrix,
                              eval_row, parameters)
            if bmagn < 0:
                bmagn /= self.afm_factor
                for dof_idx in range(dof.shape[0]):
                    bmagn_prime[dof_idx] /= self.afm_factor
            if curie_temp < 0:
                curie_temp /= self.afm_factor
                for dof_idx in range(dof.shape[0]):
                    bmagn_prime[dof_idx] /= self.afm_factor
            if curie_temp > 1e-6:
                tau = dof[1] / curie_temp
                for dof_idx in range(dof.shape[0]):
                    if dof_idx == 1:
                        # wrt T
                        tau_prime[1] = (curie_temp - dof[1]*curie_temp_prime[1])/(curie_temp**2)
                    else:
                        tau_prime[dof_idx] = -dof[1] * curie_temp_prime[dof_idx]/(curie_temp**2)
                # factor when tau < 1
                if tau < 1:
                    g_func = 1 - (1./A) * ((79./(140*p))*(tau**(-1)) + (474./497)*(1./p - 1) \
                        * ((tau**3)/6 + (tau**9)/135 + (tau**15)/600)
                                      )
                    g_func_prime = (1./A)*((79./(140*p)) / (tau**2) - (474./497)*(1./p - 1)*(tau**2 / 2 \
                        + tau**14 / 40 + tau**8 / 15))
                else:
                    # factor when tau >= 1
                    g_func = -(1./A) * ((tau**-5)/10 + (tau**-15)/315. + (tau**-25)/1500.)
                    g_func_prime = (1./A) * (1./(60*tau**26) + 1./(21*tau**16) + 1./(2*tau**6))
                for dof_idx in range(dof.shape[0]):
                    if dof_idx != 1:
                        out[dof_idx] += 8.3145 * dof[1] * (bmagn_prime[dof_idx] * g_func / (bmagn+1) + \
                                                           log(bmagn+1) * tau_prime[dof_idx] * g_func_prime)
                    else:
                        # wrt T
                        out[dof_idx] += 8.3145 * (((dof[1] * bmagn_prime[dof_idx]) / (bmagn+1) + log(bmagn+1)) * g_func + \
                            dof[1] * log(bmagn+1) * tau_prime[dof_idx] * g_func_prime)

        for subl_idx in range(self.site_ratios.shape[0]):
            if (self.vacancy_index > -1) and self.composition_matrices[self.vacancy_index, subl_idx, 1] > -1:
                mass_normalization_factor += self.site_ratios[subl_idx] * (1-dof[2+<int>self.composition_matrices[self.vacancy_index, subl_idx, 1]])
                mass_normalization_vacancy_factor[<int>self.composition_matrices[self.vacancy_index, subl_idx, 1]] = -self.site_ratios[subl_idx]
                if energy[0] == 0:
                    self.eval_energy(energy, np.array([dof]), parameters)
            else:
                mass_normalization_factor += self.site_ratios[subl_idx]
        for dof_idx in range(2+sum(self.sublattice_dof)):
            if (dof_idx > 1) and out[dof_idx] != 0 and mass_normalization_vacancy_factor[dof_idx-2] != 0:
                out[dof_idx] = (out[dof_idx]/mass_normalization_factor) - (energy[0] * mass_normalization_vacancy_factor[dof_idx-2]) / (mass_normalization_factor**2)
            else:
                out[dof_idx] /= mass_normalization_factor
        for dof_idx in range(2+sum(self.sublattice_dof)):
            out_grad[dof_idx] = out_grad[dof_idx] + sign * out[dof_idx]

    cpdef eval_energy_gradient(self, double[::1] out, double[:] dof, double[:] parameters):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] eval_row = np.zeros(4+sum(self.sublattice_dof))
        cdef np.ndarray[ndim=1, dtype=np.float64_t] disordered_eval_row
        cdef np.ndarray[ndim=2, dtype=np.float64_t] disordered_dof
        cdef np.ndarray[ndim=2, dtype=np.float64_t] ordered_dof
        cdef np.ndarray[ndim=1, dtype=np.float64_t]  row
        cdef double disordered_mass_normalization_factor = 0
        cdef double disordered_curie_temp = 0
        cdef double tau = 0
        cdef double disordered_bmagn = 0
        cdef double A, p, res_tau
        cdef double disordered_energy = 0
        cdef int prev_idx = 0
        cdef int dof_idx, out_idx, subl_idx
        self._eval_energy_gradient(out, dof, parameters, 1)
        if self.ordered:
            raise NotImplementedError
        if self._debug:
            debugout = np.zeros_like(out)
            self._debug_energy_gradient(debugout, np.asfortranarray(dof), np.ascontiguousarray(parameters))
            try:
                np.testing.assert_allclose(out,debugout)
            except AssertionError as e:
                print('--')
                print(e)
                print(np.array(debugout)-np.array(out))
                print('DOF:', np.array(dof))
                print(self.constituents)
                print('--')