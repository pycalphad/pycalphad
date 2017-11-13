import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log
from libc.stdlib cimport calloc, malloc, free
from libc.string cimport memset
from sympy import Symbol
from tinydb import where
from collections import OrderedDict
from pycalphad.core.rksum import RedlichKisterSum
from pycalphad.core.sympydiff_utils import build_functions
from pycalphad.core.constants import BIGNUM
import pycalphad.variables as v
from pycalphad import Model
from pycalphad.model import DofError
from cpython cimport PyCapsule_CheckExact, PyCapsule_GetPointer
from pickle import PicklingError


cdef double MAX_ENERGY = BIGNUM

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
cdef void* cython_pointer(obj):
    if (PyCapsule_CheckExact(obj)):
        return PyCapsule_GetPointer(obj, NULL)
    raise ValueError("Not an object containing a void ptr")

# Forward declaration necessary for some self-referencing below
cdef public class CompiledModel(object)[type CompiledModelType, object CompiledModelObject]

cdef public class CompiledModel(object)[type CompiledModelType, object CompiledModelObject]:
    def __init__(self, dbe, comps, phase_name, parameters=None, _debug=False):
        cdef int subl_index, comp_index
        possible_comps = set([x.upper() for x in comps])
        comps = sorted(comps, key=str)
        phase = dbe.phases[phase_name]
        self.phase_name = <unicode>phase_name
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
        parameters = parameters or dict()
        new_parameters = OrderedDict()
        for param, val in sorted(parameters.items(), key=str):
            if not isinstance(param, Symbol):
                new_parameters[Symbol(param)] = val
            else:
                new_parameters[param] = val
        parameters = new_parameters
        param_symbols = tuple(parameters.keys())
        self._debug = _debug
        if _debug:
            debugmodel = Model(dbe, comps, phase_name, parameters)
            out = debugmodel.energy
            undefs = list(out.atoms(Symbol) - out.atoms(v.StateVariable) - set(param_symbols))
            for undef in undefs:
                out = out.xreplace({undef: float(0)})
            _debugobj, _debuggrad, _debughess = build_functions(out, tuple([v.P, v.T] + self.variables),
                                                                parameters=param_symbols)
            # Trigger lazy computation
            if _debugobj is not None:
                _debugobj.kernel
                self._debugobj = <func_t*> cython_pointer(_debugobj._cpointer)
            if _debuggrad is not None:
                _debuggrad.kernel
                self._debuggrad = <func_novec_t*> cython_pointer(_debuggrad._cpointer)
            if _debughess is not None:
                _debughess.kernel
                self._debughess = <func_novec_t*> cython_pointer(_debughess._cpointer)

        self.site_ratios = np.array([float(x) for x in phase.sublattices])
        self.sublattice_dof = np.array([len(c) for c in self.constituents], dtype=np.int32)
        self.phase_dof = sum(self.sublattice_dof)
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
            self.disordered_phase_dof = sum(self.disordered_sublattice_dof)
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
            self.disordered_sublattice_dof = np.array([], dtype=np.int32)
            self.disordered_phase_dof = 0
            self.disordered_site_ratios = np.array([])
            self.disordered_composition_matrices = np.ascontiguousarray(np.atleast_3d([]))
            self.disordered_pure_coef_matrix = np.atleast_2d([])
            self.disordered_pure_coef_symbol_matrix = np.atleast_2d([])
            self.disordered_excess_coef_matrix = np.atleast_2d([])
            self.disordered_excess_coef_symbol_matrix = np.atleast_2d([])
            self.disordered_bm_coef_matrix = np.atleast_2d([])
            self.disordered_bm_coef_symbol_matrix = np.atleast_2d([])
            self.disordered_tc_coef_matrix = np.atleast_2d([])
            self.disordered_tc_coef_symbol_matrix = np.atleast_2d([])
            self.disordered_ihj_magnetic_structure_factor = 0
            self.disordered_afm_factor = 0

    def __reduce__(self):
        if self._debug:
            raise PicklingError('Cannot pickle CompiledModel in debug mode')
        return (_rebuild_compiledmodel, (self.constituents, self.variables, self.components,
                                         np.asarray(self.sublattice_dof), self.phase_dof,
                                         np.asarray(self.composition_matrices),
                                         np.asarray(self.site_ratios), np.asarray(self.vacancy_index),
                                         np.asarray(self.pure_coef_matrix), np.asarray(self.pure_coef_symbol_matrix),
                                         np.asarray(self.excess_coef_matrix), np.asarray(self.excess_coef_symbol_matrix),
                                         np.asarray(self.bm_coef_matrix), np.asarray(self.bm_coef_symbol_matrix),
                                         np.asarray(self.tc_coef_matrix), np.asarray(self.tc_coef_symbol_matrix),
                                         self.ihj_magnetic_structure_factor, self.afm_factor,
                                         np.asarray(self.disordered_sublattice_dof),
                                         self.disordered_phase_dof,
                                         np.asarray(self.disordered_composition_matrices),
                                         np.asarray(self.disordered_site_ratios),
                                         np.asarray(self.disordered_pure_coef_matrix),
                                         np.asarray(self.disordered_pure_coef_symbol_matrix),
                                         np.asarray(self.disordered_excess_coef_matrix),
                                         np.asarray(self.disordered_excess_coef_symbol_matrix),
                                         np.asarray(self.disordered_bm_coef_matrix),
                                         np.asarray(self.disordered_bm_coef_symbol_matrix),
                                         np.asarray(self.disordered_tc_coef_matrix),
                                         np.asarray(self.disordered_tc_coef_symbol_matrix),
                                         self.disordered_ihj_magnetic_structure_factor,
                                         self.disordered_afm_factor, self.ordered, self._debug))

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
                                double *eval_row, double[:] parameters) nogil:
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

    @cython.boundscheck(False)
    cdef void _eval_rk_matrix_gradient(self, double *out, double[:,:] coef_mat, double[:,:] symbol_mat,
                                       double *eval_row, double[:] parameters) nogil:
        cdef double result = 0
        cdef double prod_result
        cdef int row_idx1 = 0
        cdef int row_idx2 = 0
        cdef int col_idx = 0
        cdef int dof_idx
        # Either of these matrices could be empty; check both to be safe
        cdef int dof_len = max(coef_mat.shape[1]-6, symbol_mat.shape[1]-6)
        # eval_row order: P,T,ln(P),ln(T),y...
        # dof order: P,T,y...
        # coef_mat order: low_temp,high_temp,P,T,ln(P),ln(T),y...,constant_term,parameter_value
        for dof_idx in range(dof_len):
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
                            if (symbol_mat[row_idx2, 2+dof_idx] != 0) or (symbol_mat[row_idx2, 4+dof_idx] != 0):
                                prod_result = symbol_mat[row_idx2, symbol_mat.shape[1]-2] * parameters[<int>symbol_mat[row_idx2, symbol_mat.shape[1]-1]]
                                prod_result *= eval_row[dof_idx] ** (symbol_mat[row_idx2, 2+dof_idx] - 1)
                                prod_result *= eval_row[2+dof_idx] ** (symbol_mat[row_idx2, 4+dof_idx] - 1)
                                prod_result *= symbol_mat[row_idx2, 2+dof_idx] * eval_row[2+dof_idx] + symbol_mat[row_idx2, 4+dof_idx]
                                for col_idx in range(4, symbol_mat.shape[1]-4):
                                        prod_result *= (eval_row[col_idx] ** symbol_mat[row_idx2, 2+col_idx])
                                out[dof_idx] += prod_result
                        else:
                            if symbol_mat[row_idx2, 4+dof_idx] != 0:
                                prod_result = symbol_mat[row_idx2, symbol_mat.shape[1]-2] * parameters[<int>symbol_mat[row_idx2, symbol_mat.shape[1]-1]]
                                for col_idx in range(symbol_mat.shape[1]-4):
                                    if col_idx == 2+dof_idx:
                                        prod_result *= (symbol_mat[row_idx2, 2+col_idx] * eval_row[col_idx] ** (symbol_mat[row_idx2, 2+col_idx] - 1))
                                    else:
                                        prod_result *= (eval_row[col_idx] ** symbol_mat[row_idx2, 2+col_idx])
                                out[dof_idx] += prod_result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _eval_energy(self, double *out, double *dof, double[:] parameters, double sign, size_t num_pts) nogil:
        cdef int eval_row_len = 4+self.phase_dof
        # This 2-D array will be C-ordered
        cdef double *eval_row = <double*>malloc(num_pts*eval_row_len * sizeof(double))
        cdef double *mass_normalization_factor = <double*>malloc(num_pts * sizeof(double))
        cdef double *curie_temp = <double*>malloc(num_pts * sizeof(double))
        cdef double *tau = <double*>malloc(num_pts * sizeof(double))
        cdef double *bmagn = <double*>malloc(num_pts * sizeof(double))
        cdef double *res_tau = <double*>malloc(num_pts * sizeof(double))
        cdef double p = self.ihj_magnetic_structure_factor
        cdef double A = 518./1125 + (11692./15975)*(1./p - 1.)
        cdef double *out_energy = <double*>malloc(num_pts * sizeof(double))
        cdef int *prev_idx = <int*>malloc(num_pts * sizeof(int))
        cdef size_t dof_idx, out_idx, eval_idx
        cdef size_t dof_len = 2 + self.phase_dof
        for out_idx in range(num_pts):
            out_energy[out_idx] = 0
            mass_normalization_factor[out_idx] = 0
            eval_row[eval_row_len*out_idx + 0] = dof[dof_len*out_idx + 0]
            eval_row[eval_row_len*out_idx + 1] = dof[dof_len*out_idx + 1]
            eval_row[eval_row_len*out_idx + 2] = log(dof[dof_len*out_idx + 0])
            eval_row[eval_row_len*out_idx + 3] = log(dof[dof_len*out_idx + 1])
            for eval_idx in range(self.phase_dof):
                eval_row[eval_row_len*out_idx + 4+eval_idx] = dof[dof_len*out_idx + 2+eval_idx]
            # Ideal mixing
            prev_idx[out_idx] = 0
            for entry_idx in range(self.site_ratios.shape[0]):
                for dof_idx in range(prev_idx[out_idx], prev_idx[out_idx]+self.sublattice_dof[entry_idx]):
                    if dof[dof_len*out_idx + 2+dof_idx] > 1e-16:
                        out_energy[out_idx] += 8.3145 * dof[dof_len*out_idx + 1] * self.site_ratios[entry_idx] * dof[dof_len*out_idx + 2+dof_idx] * log(dof[dof_len*out_idx + 2+dof_idx])
                prev_idx[out_idx] += self.sublattice_dof[entry_idx]

            # End-member contribution
            out_energy[out_idx] += self._eval_rk_matrix(self.pure_coef_matrix, self.pure_coef_symbol_matrix,
                                                        eval_row+eval_row_len*out_idx, parameters)
            # Interaction contribution
            out_energy[out_idx] += self._eval_rk_matrix(self.excess_coef_matrix, self.excess_coef_symbol_matrix,
                                                        eval_row+eval_row_len*out_idx, parameters)
            # Magnetic contribution
            curie_temp[out_idx] = self._eval_rk_matrix(self.tc_coef_matrix, self.tc_coef_symbol_matrix,
                                                       eval_row+eval_row_len*out_idx, parameters)
            bmagn[out_idx] = self._eval_rk_matrix(self.bm_coef_matrix, self.bm_coef_symbol_matrix,
                                                  eval_row+eval_row_len*out_idx, parameters)
            if (curie_temp[out_idx] != 0) and (bmagn[out_idx] != 0) and (self.ihj_magnetic_structure_factor > 0) and (self.afm_factor != 0):
                if bmagn[out_idx] < 0:
                    bmagn[out_idx] /= self.afm_factor
                if curie_temp[out_idx] < 0:
                    curie_temp[out_idx] /= self.afm_factor
                if curie_temp[out_idx] > 1e-6:
                    tau[out_idx] = dof[dof_len*out_idx + 1] / curie_temp[out_idx]
                    # factor when tau < 1
                    if tau[out_idx] < 1:
                        res_tau[out_idx] = 1 - (1./A) * ((79./(140*p))*(tau[out_idx]**(-1)) + (474./497)*(1./p - 1) \
                            * ((tau[out_idx]**3)/6 + (tau[out_idx]**9)/135 + (tau[out_idx]**15)/600)
                                          )
                    else:
                        # factor when tau >= 1
                        res_tau[out_idx] = -(1/A) * ((tau[out_idx]**-5)/10 + (tau[out_idx]**-15)/315. + (tau[out_idx]**-25)/1500.)
                    out_energy[out_idx] += 8.3145 * dof[dof_len*out_idx + 1] * log(bmagn[out_idx]+1) * res_tau[out_idx]
            for subl_idx in range(self.site_ratios.shape[0]):
                if (self.vacancy_index > -1) and self.composition_matrices[self.vacancy_index, subl_idx, 1] > -1:
                    mass_normalization_factor[out_idx] += self.site_ratios[subl_idx] * (1-dof[dof_len*out_idx + 2+<size_t>self.composition_matrices[self.vacancy_index, subl_idx, 1]])
                else:
                    mass_normalization_factor[out_idx] += self.site_ratios[subl_idx]
            if mass_normalization_factor[out_idx] <= 1e-6:
                out_energy[out_idx] = MAX_ENERGY
            else:
                out_energy[out_idx] /= mass_normalization_factor[out_idx]
            out[out_idx] = out[out_idx] + sign * out_energy[out_idx]
        free(eval_row)
        free(mass_normalization_factor)
        free(curie_temp)
        free(tau)
        free(bmagn)
        free(res_tau)
        free(out_energy)
        free(prev_idx)

    @cython.boundscheck(False)
    cdef void _eval_disordered_energy(self, double *out, double *dof, double[:] parameters, double sign) nogil:
        cdef double* eval_row = <double*>malloc((4+self.disordered_phase_dof) * sizeof(double))
        cdef double mass_normalization_factor = 0
        cdef double curie_temp = 0
        cdef double tau = 0
        cdef double bmagn = 0
        cdef double res_tau = 0
        cdef double p = self.disordered_ihj_magnetic_structure_factor
        cdef double A = 518./1125 + (11692./15975)*(1./p - 1.)
        cdef double out_energy = 0
        cdef int prev_idx = 0
        cdef int dof_idx, eval_idx
        eval_row[0] = dof[0]
        eval_row[1] = dof[1]
        eval_row[2] = log(dof[0])
        eval_row[3] = log(dof[1])
        for eval_idx in range(self.disordered_phase_dof):
            eval_row[4+eval_idx] = dof[2+eval_idx]
        # Ideal mixing
        prev_idx = 0
        for entry_idx in range(self.disordered_site_ratios.shape[0]):
            for dof_idx in range(prev_idx, prev_idx+self.disordered_sublattice_dof[entry_idx]):
                if dof[2+dof_idx] > 1e-16:
                    out_energy += 8.3145 * dof[1] * self.disordered_site_ratios[entry_idx] * dof[2+dof_idx] * log(dof[2+dof_idx])
            prev_idx += self.disordered_sublattice_dof[entry_idx]

        # End-member contribution
        out_energy += self._eval_rk_matrix(self.disordered_pure_coef_matrix, self.disordered_pure_coef_symbol_matrix,
                                           eval_row, parameters)
        # Interaction contribution
        out_energy += self._eval_rk_matrix(self.disordered_excess_coef_matrix, self.disordered_excess_coef_symbol_matrix,
                                           eval_row, parameters)
        # Magnetic contribution
        curie_temp = self._eval_rk_matrix(self.disordered_tc_coef_matrix, self.disordered_tc_coef_symbol_matrix,
                                          eval_row, parameters)
        bmagn = self._eval_rk_matrix(self.disordered_bm_coef_matrix, self.disordered_bm_coef_symbol_matrix,
                                     eval_row, parameters)
        if (curie_temp != 0) and (bmagn != 0) and (self.disordered_ihj_magnetic_structure_factor > 0) and (self.disordered_afm_factor != 0):
            if bmagn < 0:
                bmagn /= self.disordered_afm_factor
            if curie_temp < 0:
                curie_temp /= self.disordered_afm_factor
            if curie_temp > 1e-6:
                tau = dof[1] / curie_temp
                # factor when tau < 1
                if tau < 1:
                    res_tau = 1 - (1./A) * ((79./(140*p))*(tau**(-1)) + (474./497)*(1./p - 1) \
                        * ((tau**3)/6 + (tau**9)/135 + (tau**15)/600)
                                      )
                else:
                    # factor when tau >= 1
                    res_tau = -(1/A) * ((tau**-5)/10 + (tau**-15)/315. + (tau**-25)/1500.)
                out_energy += 8.3145 * dof[1] * log(bmagn+1) * res_tau
        for subl_idx in range(self.disordered_site_ratios.shape[0]):
            if (self.vacancy_index > -1) and self.disordered_composition_matrices[self.vacancy_index, subl_idx, 1] > -1:
                mass_normalization_factor += self.disordered_site_ratios[subl_idx] * (1-dof[2+<int>self.disordered_composition_matrices[self.vacancy_index, subl_idx, 1]])
            else:
                mass_normalization_factor += self.disordered_site_ratios[subl_idx]
        if mass_normalization_factor <= 1e-6:
            out_energy = MAX_ENERGY
        else:
            out_energy /= mass_normalization_factor
        out[0] = out[0] + sign * out_energy
        free(eval_row)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _compute_disordered_dof(self, double *disordered_dof, double *dof, size_t num_pts) nogil:
        cdef size_t out_idx, dof_idx, comp_idx, subl_idx, disordered_dof_idx, copy_idx
        cdef int num_comps
        cdef double site_sum
        cdef size_t disordered_dof_len = 2 + self.disordered_phase_dof
        cdef size_t dof_len = 2 + self.phase_dof
        dof_idx = _intsum(self.sublattice_dof[:self.sublattice_dof.shape[0]-1])
        disordered_dof_idx = _intsum(self.disordered_sublattice_dof[:self.disordered_sublattice_dof.shape[0]-1])
        for out_idx in range(num_pts):
            site_sum = 0
            num_comps = 0
            # Disordered phase contribution
            # Assume: Same components in all sublattices, except maybe a pure VA sublattice at the end
            disordered_dof[disordered_dof_len*out_idx + 0] = dof[dof_len*out_idx + 0]
            disordered_dof[disordered_dof_len*out_idx + 1] = dof[dof_len*out_idx + 1]
            num_comps = self.sublattice_dof[0]
            # Last sublattice is different from first; probably an interstitial sublattice
            # It should be treated separately
            if self.sublattice_dof[0] != self.sublattice_dof[self.sublattice_dof.shape[0]-1]:
                site_sum = _sum(self.site_ratios[:self.site_ratios.shape[0]-1])
                for subl_idx in range(self.site_ratios.shape[0]-1):
                    for comp_idx in range(self.sublattice_dof[subl_idx]):
                        disordered_dof[disordered_dof_len*out_idx + comp_idx+2] += (self.site_ratios[subl_idx] / site_sum) * dof[dof_len*out_idx + (subl_idx * num_comps + comp_idx + 2)]

                # Copy interstitial values directly
                for copy_idx in range(self.disordered_sublattice_dof[self.disordered_sublattice_dof.shape[0]-1]):
                    disordered_dof[disordered_dof_len*out_idx + disordered_dof_idx+2+copy_idx] = dof[dof_len*out_idx + dof_idx+2+copy_idx]
            else:
                site_sum = _sum(self.site_ratios)
                for subl_idx in range(self.site_ratios.shape[0]):
                    for comp_idx in range(self.sublattice_dof[subl_idx]):
                        disordered_dof[disordered_dof_len*out_idx + subl_idx+2] += (self.site_ratios[subl_idx] / site_sum) * dof[dof_len*out_idx + (subl_idx * num_comps + comp_idx + 2)]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _compute_ordered_dof(self, double *ordered_dof, double *disordered_dof, size_t num_pts) nogil:
        cdef size_t dof_idx, out_idx, subl_idx, comp_idx, disordered_dof_idx, copy_idx
        cdef int num_comps = self.sublattice_dof[0]
        cdef size_t ordered_dof_len = 2 + self.phase_dof
        cdef size_t disordered_dof_len = 2 + self.disordered_phase_dof
        dof_idx = _intsum(self.sublattice_dof[:self.sublattice_dof.shape[0]-1])
        disordered_dof_idx = _intsum(self.disordered_sublattice_dof[:self.disordered_sublattice_dof.shape[0]-1])
        for out_idx in range(num_pts):
            # Subtract ordered energy at disordered configuration
            ordered_dof[ordered_dof_len*out_idx + 0] = disordered_dof[disordered_dof_len*out_idx + 0]
            ordered_dof[ordered_dof_len*out_idx + 1] = disordered_dof[disordered_dof_len*out_idx + 1]
            if self.sublattice_dof[0] != self.sublattice_dof[self.sublattice_dof.shape[0]-1]:
                for subl_idx in range(self.site_ratios.shape[0]-1):
                    for comp_idx in range(self.sublattice_dof[subl_idx]):
                        ordered_dof[ordered_dof_len*out_idx + (subl_idx * num_comps + comp_idx + 2)] = disordered_dof[disordered_dof_len*out_idx + comp_idx+2]

                # Copy interstitial values directly
                for copy_idx in range(self.sublattice_dof[self.sublattice_dof.shape[0]-1]):
                    ordered_dof[ordered_dof_len*out_idx + dof_idx+2+copy_idx] = disordered_dof[disordered_dof_len*out_idx + disordered_dof_idx+2+copy_idx]
            else:
                for subl_idx in range(self.site_ratios.shape[0]):
                    for comp_idx in range(self.sublattice_dof[subl_idx]):
                        ordered_dof[ordered_dof_len*out_idx + (subl_idx * num_comps + comp_idx + 2)] = disordered_dof[disordered_dof_len*out_idx + subl_idx+2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void eval_energy(self, double *out, double *dof, double[:] parameters, size_t num_pts) nogil:
        # 1-D
        cdef double *disordered_eval_row = NULL
        # 2-D C-ordered
        cdef double *disordered_dof = NULL
        # 2-D C-ordered
        cdef double *ordered_dof = NULL
        cdef double disordered_mass_normalization_factor = 0
        cdef double disordered_curie_temp = 0
        cdef double tau = 0
        cdef double disordered_bmagn = 0
        cdef double A, p, res_tau
        cdef double disordered_energy = 0
        cdef size_t prev_idx = 0
        cdef size_t dof_idx, out_idx, subl_idx, disordered_dof_idx
        cdef size_t disordered_dof_len = 0
        cdef size_t ordered_dof_len = 0
        self._eval_energy(out, dof, parameters, 1, num_pts)
        if self.ordered:
            disordered_dof_len = 2 + self.disordered_phase_dof
            ordered_dof_len = 2 + self.phase_dof
            disordered_dof = <double*>calloc(num_pts * disordered_dof_len, sizeof(double))
            ordered_dof = <double*>calloc(num_pts * ordered_dof_len, sizeof(double))
            self._compute_disordered_dof(disordered_dof, dof, num_pts)
            self._compute_ordered_dof(ordered_dof, disordered_dof, num_pts)
            # Subtract ordered energy at disordered configuration
            self._eval_energy(out, ordered_dof, parameters, -1, num_pts)
            disordered_eval_row = <double*>malloc((self.disordered_phase_dof+4) * sizeof(double))
            for out_idx in range(num_pts):
                disordered_mass_normalization_factor = 0
                disordered_curie_temp = 0
                disordered_bmagn = 0
                disordered_energy = 0
                # Disordered phase contribution
                # Assume: Same components in all sublattices, except maybe a pure VA sublattice at the end
                disordered_eval_row[0] = disordered_dof[disordered_dof_len*out_idx + 0]
                disordered_eval_row[1] = disordered_dof[disordered_dof_len*out_idx + 1]
                disordered_eval_row[2] = log(disordered_dof[disordered_dof_len*out_idx + 0])
                disordered_eval_row[3] = log(disordered_dof[disordered_dof_len*out_idx + 1])
                for disordered_dof_idx in range(self.disordered_phase_dof):
                    disordered_eval_row[4+disordered_dof_idx] = disordered_dof[disordered_dof_len*out_idx + 2+disordered_dof_idx]
                # Ideal mixing
                prev_idx = 0
                for entry_idx in range(self.disordered_site_ratios.shape[0]):
                    for dof_idx in range(prev_idx, prev_idx+self.disordered_sublattice_dof[entry_idx]):
                        if disordered_dof[disordered_dof_len*out_idx + 2+dof_idx] > 1e-16:
                            disordered_energy += 8.3145 * disordered_dof[disordered_dof_len*out_idx + 1] * self.disordered_site_ratios[entry_idx] * disordered_dof[disordered_dof_len*out_idx + 2+dof_idx] * log(disordered_dof[disordered_dof_len*out_idx + 2+dof_idx])
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
                        tau = dof[ordered_dof_len*out_idx + 1] / disordered_curie_temp
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
                        disordered_energy += 8.3145 * disordered_dof[disordered_dof_len*out_idx + 1] * log(disordered_bmagn+1) * res_tau
                for subl_idx in range(self.disordered_site_ratios.shape[0]):
                    if (self.vacancy_index > -1) and self.disordered_composition_matrices[self.vacancy_index, subl_idx, 1] > -1:
                        disordered_mass_normalization_factor += self.disordered_site_ratios[subl_idx] * (1-disordered_dof[disordered_dof_len*out_idx + 2+<size_t>self.disordered_composition_matrices[self.vacancy_index, subl_idx, 1]])
                    else:
                        disordered_mass_normalization_factor += self.disordered_site_ratios[subl_idx]
                if disordered_mass_normalization_factor <= 1e-6:
                    disordered_energy = MAX_ENERGY
                else:
                    disordered_energy /= disordered_mass_normalization_factor
                out[out_idx] += disordered_energy
            free(disordered_eval_row)
            free(disordered_dof)
            free(ordered_dof)
        if self._debug:
            with gil:
                debugout = np.zeros(num_pts)
                self._debug_energy(debugout, np.ascontiguousarray(<double[:num_pts,:2+self.phase_dof]>dof), np.ascontiguousarray(parameters))
                np.testing.assert_allclose(<double[:num_pts]>out,debugout)

    cdef _debug_energy(self, double[::1] debugout, double[:,::1] dof, double[::1] parameters):
        if parameters.shape[0] == 0:
            self._debugobj(&debugout[0], &dof[0,0], NULL, debugout.shape[0])
        else:
            self._debugobj(&debugout[0], &dof[0,0], &parameters[0], debugout.shape[0])

    cdef _debug_energy_gradient(self, double[::1] debugout, double[::1] dof, double[::1] parameters):
        if parameters.shape[0] == 0:
            self._debuggrad(&dof[0], NULL, &debugout[0])
        else:
            self._debuggrad(&dof[0], &parameters[0], &debugout[0])

    @cython.boundscheck(False)
    cdef void _eval_energy_gradient(self, double *out_grad, double *dof, double[:] parameters, double sign) nogil:
        # This 2-D array will be C-ordered
        cdef size_t dof_len = 2 + self.phase_dof
        cdef double *eval_row = <double*>calloc((4+self.phase_dof), sizeof(double))
        cdef double *out = <double*>calloc(dof_len, sizeof(double))
        cdef double energy = 0
        cdef double mass_normalization_factor = 0
        cdef double *mass_normalization_vacancy_factor = <double*>calloc(self.phase_dof, sizeof(double))
        cdef double curie_temp = 0
        cdef double *curie_temp_prime = <double*>calloc((2+self.phase_dof), sizeof(double))
        cdef double tau = 0
        cdef double *tau_prime = <double*>calloc((2+self.phase_dof), sizeof(double))
        cdef double bmagn = 0
        cdef double *bmagn_prime = <double*>calloc((2+self.phase_dof), sizeof(double))
        cdef double g_func = 0
        cdef double g_func_prime = 0
        cdef double p
        cdef double A
        cdef int prev_idx = 0
        cdef int dof_idx, eval_idx

        eval_row[0] = dof[0]
        eval_row[1] = dof[1]
        eval_row[2] = log(dof[0])
        eval_row[3] = log(dof[1])
        for eval_idx in range(self.phase_dof):
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
            p = self.ihj_magnetic_structure_factor
            A = 518./1125 + (11692./15975)*(1./p - 1.)
            self._eval_rk_matrix_gradient(curie_temp_prime, self.tc_coef_matrix, self.tc_coef_symbol_matrix,
                              eval_row, parameters)
            self._eval_rk_matrix_gradient(bmagn_prime, self.bm_coef_matrix, self.bm_coef_symbol_matrix,
                              eval_row, parameters)
            if bmagn < 0:
                bmagn /= self.afm_factor
                for dof_idx in range(2+self.phase_dof):
                    bmagn_prime[dof_idx] /= self.afm_factor
            if curie_temp < 0:
                curie_temp /= self.afm_factor
                for dof_idx in range(2+self.phase_dof):
                    curie_temp_prime[dof_idx] /= self.afm_factor
            if curie_temp > 1e-6:
                tau = dof[1] / curie_temp
                for dof_idx in range(2+self.phase_dof):
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
                for dof_idx in range(dof_len):
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
                if energy == 0:
                    self.eval_energy(&energy, &dof[0], parameters, 1)
            else:
                mass_normalization_factor += self.site_ratios[subl_idx]
        if mass_normalization_factor <= 1e-6:
            out[dof_idx] = MAX_ENERGY
        else:
            for dof_idx in range(2+self.phase_dof):
                if (dof_idx > 1) and out[dof_idx] != 0 and mass_normalization_vacancy_factor[dof_idx-2] != 0:
                    # Remember that energy is already equal to the energy divided by the mass normalization factor
                    # That is why one factor of it disappears in the formula
                    out[dof_idx] = (out[dof_idx]/mass_normalization_factor) - (energy * mass_normalization_vacancy_factor[dof_idx-2]) / mass_normalization_factor
                else:
                    out[dof_idx] /= mass_normalization_factor
        for dof_idx in range(2+self.phase_dof):
            out_grad[dof_idx] = out_grad[dof_idx] + sign * out[dof_idx]
        free(eval_row)
        free(out)
        free(mass_normalization_vacancy_factor)
        free(curie_temp_prime)
        free(tau_prime)
        free(bmagn_prime)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void eval_energy_gradient(self, double *out, double *dof, double[:] parameters) nogil:
        # 1-D
        cdef double *disordered_eval_row = NULL
        # 2-D C-ordered
        cdef double *disordered_dof = NULL
        # 1-D
        cdef double *disordered_out = NULL
        # 2-D C-ordered
        cdef double *ordered_dof = NULL
        cdef double *disordered_mass_normalization_vacancy_factor
        cdef double disordered_curie_temp = 0
        cdef double *disordered_curie_temp_prime
        cdef double disordered_tau = 0
        cdef double *disordered_tau_prime
        cdef double disordered_bmagn = 0
        cdef double *disordered_bmagn_prime
        cdef double disordered_g_func = 0
        cdef double disordered_g_func_prime = 0
        cdef double disordered_p
        cdef double disordered_A
        cdef double disordered_mass_normalization_factor = 0
        cdef double disordered_energy = 0
        cdef int prev_idx = 0
        cdef int dof_idx, out_idx, subl_idx, eval_idx, disordered_dof_idx
        cdef size_t disordered_dof_len = 0
        self._eval_energy_gradient(out, dof, parameters, 1)
        if self.ordered:
            disordered_dof_len = 2 + self.disordered_phase_dof
            disordered_out = <double*>calloc(disordered_dof_len, sizeof(double))
            disordered_dof = <double*>calloc(disordered_dof_len, sizeof(double))
            ordered_dof = <double*>calloc(2+self.phase_dof, sizeof(double))
            self._compute_disordered_dof(disordered_dof, &dof[0], 1)
            self._compute_ordered_dof(ordered_dof, disordered_dof, 1)
            # Subtract ordered energy gradient at disordered configuration
            self._eval_energy_gradient(out, ordered_dof, parameters, -1)
            # Add disordered energy gradient
            disordered_eval_row = <double*>malloc((4+self.disordered_phase_dof) * sizeof(double))
            disordered_mass_normalization_factor = 0
            disordered_mass_normalization_vacancy_factor = <double*>calloc(_intsum(self.disordered_sublattice_dof)+2, sizeof(double))
            disordered_curie_temp = 0
            disordered_bmagn = 0
            disordered_eval_row[0] = dof[0]
            disordered_eval_row[1] = dof[1]
            disordered_eval_row[2] = log(dof[0])
            disordered_eval_row[3] = log(dof[1])
            for eval_idx in range(self.disordered_phase_dof):
                disordered_eval_row[4+eval_idx] = disordered_dof[2+eval_idx]
            # Ideal mixing
            for entry_idx in range(self.disordered_site_ratios.shape[0]):
                for dof_idx in range(prev_idx, prev_idx+self.disordered_sublattice_dof[entry_idx]):
                    if disordered_dof[2+dof_idx] > 1e-16:
                        # wrt P: 0
                        # wrt T
                        disordered_out[1] += 8.3145 * self.disordered_site_ratios[entry_idx] * disordered_dof[2+dof_idx] * log(disordered_dof[2+dof_idx])
                    # wrt y
                    disordered_out[2+dof_idx] += 8.3145 * disordered_dof[1] * self.disordered_site_ratios[entry_idx] * (log(disordered_dof[2+dof_idx]) + 1)
                prev_idx += self.disordered_sublattice_dof[entry_idx]

            # End-member contribution
            self._eval_rk_matrix_gradient(disordered_out, self.disordered_pure_coef_matrix, self.disordered_pure_coef_symbol_matrix,
                                          disordered_eval_row, parameters)
            # Interaction contribution
            self._eval_rk_matrix_gradient(disordered_out, self.disordered_excess_coef_matrix, self.disordered_excess_coef_symbol_matrix,
                                          disordered_eval_row, parameters)
            # Magnetic contribution
            disordered_curie_temp = self._eval_rk_matrix(self.disordered_tc_coef_matrix, self.disordered_tc_coef_symbol_matrix,
                                                         disordered_eval_row, parameters)
            disordered_bmagn = self._eval_rk_matrix(self.disordered_bm_coef_matrix, self.disordered_bm_coef_symbol_matrix,
                                                    disordered_eval_row, parameters)
            if (disordered_curie_temp != 0) and (disordered_bmagn != 0) and (self.disordered_ihj_magnetic_structure_factor > 0) and (self.disordered_afm_factor != 0):
                disordered_p = self.disordered_ihj_magnetic_structure_factor
                disordered_A = 518./1125 + (11692./15975)*(1./disordered_p - 1.)
                disordered_curie_temp_prime = <double*>calloc(2+self.disordered_phase_dof, sizeof(double))
                disordered_bmagn_prime = <double*>calloc(2+self.disordered_phase_dof, sizeof(double))
                disordered_tau_prime = <double*>calloc(2+self.disordered_phase_dof, sizeof(double))
                self._eval_rk_matrix_gradient(disordered_curie_temp_prime, self.disordered_tc_coef_matrix, self.disordered_tc_coef_symbol_matrix,
                                  disordered_eval_row, parameters)
                self._eval_rk_matrix_gradient(disordered_bmagn_prime, self.disordered_bm_coef_matrix, self.disordered_bm_coef_symbol_matrix,
                                  disordered_eval_row, parameters)
                if disordered_bmagn < 0:
                    disordered_bmagn /= self.disordered_afm_factor
                    for dof_idx in range(disordered_dof_len):
                        disordered_bmagn_prime[dof_idx] /= self.disordered_afm_factor
                if disordered_curie_temp < 0:
                    disordered_curie_temp /= self.disordered_afm_factor
                    for dof_idx in range(disordered_dof_len):
                        disordered_curie_temp_prime[dof_idx] /= self.disordered_afm_factor
                if disordered_curie_temp > 1e-6:
                    disordered_tau = disordered_dof[1] / disordered_curie_temp
                    for dof_idx in range(disordered_dof_len):
                        if dof_idx == 1:
                            # wrt T
                            disordered_tau_prime[1] = (disordered_curie_temp - disordered_dof[1]*disordered_curie_temp_prime[1])/(disordered_curie_temp**2)
                        else:
                            disordered_tau_prime[dof_idx] = -disordered_dof[1] * disordered_curie_temp_prime[dof_idx]/(disordered_curie_temp**2)
                    # factor when disordered_tau < 1
                    if disordered_tau < 1:
                        disordered_g_func = 1 - (1./disordered_A) * ((79./(140*disordered_p))*(disordered_tau**(-1)) + (474./497)*(1./disordered_p - 1) \
                            * ((disordered_tau**3)/6 + (disordered_tau**9)/135 + (disordered_tau**15)/600)
                                          )
                        disordered_g_func_prime = (1./disordered_A)*((79./(140*disordered_p)) / (disordered_tau**2) - (474./497)*(1./disordered_p - 1)*(disordered_tau**2 / 2 \
                            + disordered_tau**14 / 40 + disordered_tau**8 / 15))
                    else:
                        # factor when disordered_tau >= 1
                        disordered_g_func = -(1./disordered_A) * ((disordered_tau**-5)/10 + (disordered_tau**-15)/315. + (disordered_tau**-25)/1500.)
                        disordered_g_func_prime = (1./disordered_A) * (1./(60*disordered_tau**26) + 1./(21*disordered_tau**16) + 1./(2*disordered_tau**6))
                    for dof_idx in range(disordered_dof_len):
                        if dof_idx != 1:
                            disordered_out[dof_idx] += 8.3145 * disordered_dof[1] * (disordered_bmagn_prime[dof_idx] * disordered_g_func / (disordered_bmagn+1) + \
                                                               log(disordered_bmagn+1) * disordered_tau_prime[dof_idx] * disordered_g_func_prime)
                        else:
                            # wrt T
                            disordered_out[dof_idx] += 8.3145 * (((disordered_dof[1] * disordered_bmagn_prime[dof_idx]) / (disordered_bmagn+1) + log(disordered_bmagn+1)) * disordered_g_func + \
                                disordered_dof[1] * log(disordered_bmagn+1) * disordered_tau_prime[dof_idx] * disordered_g_func_prime)
                free(disordered_curie_temp_prime)
                free(disordered_bmagn_prime)
                free(disordered_tau_prime)

            for subl_idx in range(self.disordered_site_ratios.shape[0]):
                if (self.vacancy_index > -1) and self.disordered_composition_matrices[self.vacancy_index, subl_idx, 1] > -1:
                    disordered_mass_normalization_factor += self.disordered_site_ratios[subl_idx] * (1-disordered_dof[2+<size_t>self.disordered_composition_matrices[self.vacancy_index, subl_idx, 1]])
                    disordered_mass_normalization_vacancy_factor[<int>self.disordered_composition_matrices[self.vacancy_index, subl_idx, 1]] = -self.disordered_site_ratios[subl_idx]
                    if disordered_energy == 0:
                        self._eval_disordered_energy(&disordered_energy, disordered_dof, parameters, 1)
                else:
                    disordered_mass_normalization_factor += self.disordered_site_ratios[subl_idx]
            if disordered_mass_normalization_factor <= 1e-6:
                disordered_out[dof_idx] = MAX_ENERGY
            else:
                for dof_idx in range(disordered_dof_len):
                    if (dof_idx > 1) and disordered_out[dof_idx] != 0 and disordered_mass_normalization_vacancy_factor[dof_idx-2] != 0:
                        # Remember that disordered_energy is already equal to the energy divided by the mass normalization factor
                        # That is why one factor of it disappears in the formula
                        disordered_out[dof_idx] = (disordered_out[dof_idx]/disordered_mass_normalization_factor) - (disordered_energy * disordered_mass_normalization_vacancy_factor[dof_idx-2]) / disordered_mass_normalization_factor
                    else:
                        disordered_out[dof_idx] /= disordered_mass_normalization_factor
            # P,T derivatives can be directly added
            out[0] += disordered_out[0]
            out[1] += disordered_out[1]
            # y derivatives for disordered contribution are computed via the chain rule
            # First case: there is a different last sublattice; probably interstitial
            if self.disordered_sublattice_dof[0] != self.disordered_sublattice_dof[self.disordered_sublattice_dof.shape[0]-1]:
                for disordered_dof_idx in range(2,2+_intsum(self.disordered_sublattice_dof[:self.disordered_sublattice_dof.shape[0]-1])):
                    for subl_idx in range(self.sublattice_dof.shape[0]-1):
                        dof_idx = subl_idx * self.sublattice_dof[0] + disordered_dof_idx
                        out[dof_idx] += (self.site_ratios[subl_idx] / _sum(self.site_ratios[:self.site_ratios.shape[0]-1])) * disordered_out[disordered_dof_idx]
                # last sublattice is handled separately
                for disordered_dof_idx in range(2,2+self.disordered_sublattice_dof[self.disordered_sublattice_dof.shape[0]-1]):
                    dof_idx = (self.sublattice_dof.shape[0]-1) * self.sublattice_dof[0] + disordered_dof_idx
                    out[dof_idx] += disordered_out[_intsum(self.disordered_sublattice_dof[:self.disordered_sublattice_dof.shape[0]-1])+disordered_dof_idx]
            else:
                # Second case: all sublattices have the same degrees of freedom
                for disordered_dof_idx in range(2,2+self.disordered_phase_dof):
                    for subl_idx in range(self.sublattice_dof.shape[0]):
                        dof_idx = subl_idx * self.sublattice_dof[0] + disordered_dof_idx
                        out[dof_idx] += (self.site_ratios[subl_idx] / _sum(self.site_ratios)) * disordered_out[disordered_dof_idx]
            free(disordered_eval_row)
            free(disordered_out)
            free(disordered_mass_normalization_vacancy_factor)
            free(disordered_dof)
            free(ordered_dof)
        if self._debug:
            with gil:
                debugout = np.zeros(2+self.phase_dof)
                self._debug_energy_gradient(debugout, np.asfortranarray(<double[:2+self.phase_dof]>dof), np.ascontiguousarray(parameters))
                try:
                    np.testing.assert_allclose(np.array(<double[:2+self.phase_dof]>out),debugout)
                except AssertionError as e:
                    print('--')
                    print('Gradient mismatch')
                    print(e)
                    print(np.array(debugout)-np.array(<double[:2+self.phase_dof]>out))
                    print('DOF:', np.array(<double[:2+self.phase_dof]>dof))
                    print(self.constituents)
                    print('--')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void eval_energy_hessian(self, double[:, ::1] out, double[:] dof, double[:] parameters) nogil:
        cdef double[:,::1] debugout
        cdef double epsilon = 1e-12
        cdef size_t grad_idx
        cdef size_t col_idx
        cdef size_t total_dof = 2 + self.phase_dof
        cdef double *x1 = <double*>malloc(total_dof * sizeof(double))
        cdef double *x2 = <double*>malloc(total_dof * sizeof(double))
        cdef double *grad1 = <double*>calloc(total_dof, sizeof(double))
        cdef double *grad2 = <double*>calloc(total_dof,  sizeof(double))

        for grad_idx in range(total_dof):
            x1[grad_idx] = dof[grad_idx]
            x2[grad_idx] = dof[grad_idx]

        for grad_idx in range(total_dof):
            if grad_idx > 1:
                x1[grad_idx] = max(x1[grad_idx] - epsilon, 1e-12)
                x2[grad_idx] = min(x2[grad_idx] + epsilon, 1)
            else:
                x1[grad_idx] = x1[grad_idx] - 1e6 * epsilon
                x2[grad_idx] = x2[grad_idx] + 1e6 * epsilon
            self.eval_energy_gradient(grad1, x1, parameters)
            self.eval_energy_gradient(grad2, x2, parameters)
            for col_idx in range(total_dof):
                out[col_idx,grad_idx] = (grad2[col_idx]-grad1[col_idx])/(x2[grad_idx] - x1[grad_idx])
            x1[grad_idx] = dof[grad_idx]
            x2[grad_idx] = dof[grad_idx]
            memset(grad1, 0, total_dof * sizeof(double))
            memset(grad2, 0, total_dof * sizeof(double))
        for grad_idx in range(total_dof):
            for col_idx in range(grad_idx, total_dof):
                out[grad_idx,col_idx] = out[col_idx,grad_idx] = (out[grad_idx,col_idx]+out[col_idx,grad_idx])/2
        free(x1)
        free(x2)
        free(grad1)
        free(grad2)
        if self._debug:
            with gil:
                debugout = np.ascontiguousarray(np.zeros_like(out))
                if parameters.shape[0] == 0:
                    self._debughess(&dof[0], NULL, &debugout[0,0])
                else:
                    self._debughess(&dof[0], &parameters[0], &debugout[0,0])
                try:
                    np.testing.assert_allclose(out,debugout)
                except AssertionError as e:
                    print('--')
                    print('Hessian mismatch')
                    print(e)
                    print(np.array(debugout)-np.array(out))
                    print('DOF:', np.array(dof))
                    print(self.constituents)
                    print('--')

def _rebuild_compiledmodel(constituents, variables, components, sublattice_dof, phase_dof,
                           composition_matrices, site_ratios, vacancy_index,
                           pure_coef_matrix, pure_coef_symbol_matrix, excess_coef_matrix,
                           excess_coef_symbol_matrix, bm_coef_matrix, bm_coef_symbol_matrix,
                           tc_coef_matrix, tc_coef_symbol_matrix, ihj_magnetic_structure_factor,
                           afm_factor, disordered_sublattice_dof, disordered_phase_dof, disordered_composition_matrices,
                           disordered_site_ratios, disordered_pure_coef_matrix, disordered_pure_coef_symbol_matrix,
                           disordered_excess_coef_matrix, disordered_excess_coef_symbol_matrix,
                           disordered_bm_coef_matrix, disordered_bm_coef_symbol_matrix,
                           disordered_tc_coef_matrix, disordered_tc_coef_symbol_matrix,
                           disordered_ihj_magnetic_structure_factor, disordered_afm_factor, ordered, _debug):
    inst = CompiledModel.__new__(CompiledModel)
    (inst.constituents, inst.variables, inst.components, inst.sublattice_dof, inst.phase_dof,
    inst.composition_matrices, inst.site_ratios, inst.vacancy_index,
    inst.pure_coef_matrix, inst.pure_coef_symbol_matrix, inst.excess_coef_matrix,
    inst.excess_coef_symbol_matrix, inst.bm_coef_matrix, inst.bm_coef_symbol_matrix,
    inst.tc_coef_matrix, inst.tc_coef_symbol_matrix, inst.ihj_magnetic_structure_factor,
    inst.afm_factor, inst.disordered_sublattice_dof, inst.disordered_phase_dof, inst.disordered_composition_matrices,
    inst.disordered_site_ratios, inst.disordered_pure_coef_matrix, inst.disordered_pure_coef_symbol_matrix,
    inst.disordered_excess_coef_matrix, inst.disordered_excess_coef_symbol_matrix,
    inst.disordered_bm_coef_matrix, inst.disordered_bm_coef_symbol_matrix,
    inst.disordered_tc_coef_matrix, inst.disordered_tc_coef_symbol_matrix,
    inst.disordered_ihj_magnetic_structure_factor, inst.disordered_afm_factor, inst.ordered, inst._debug) = \
    (constituents, variables, components, sublattice_dof, phase_dof,
    composition_matrices, site_ratios, vacancy_index,
    pure_coef_matrix, pure_coef_symbol_matrix, excess_coef_matrix,
    excess_coef_symbol_matrix, bm_coef_matrix, bm_coef_symbol_matrix,
    tc_coef_matrix, tc_coef_symbol_matrix, ihj_magnetic_structure_factor,
    afm_factor, disordered_sublattice_dof, disordered_phase_dof, disordered_composition_matrices,
    disordered_site_ratios, disordered_pure_coef_matrix, disordered_pure_coef_symbol_matrix,
    disordered_excess_coef_matrix, disordered_excess_coef_symbol_matrix,
    disordered_bm_coef_matrix, disordered_bm_coef_symbol_matrix,
    disordered_tc_coef_matrix, disordered_tc_coef_symbol_matrix,
    disordered_ihj_magnetic_structure_factor, disordered_afm_factor, ordered, _debug)
    return inst
