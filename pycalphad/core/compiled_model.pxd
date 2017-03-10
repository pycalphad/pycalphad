cdef public class CompiledModel(object)[type CompiledModelType, object CompiledModelObject]:
    cdef object constituents
    cdef object variables
    cdef int[::1] sublattice_dof
    cdef double[:,:,::1] composition_matrices
    cdef double[::1] site_ratios
    cdef int vacancy_index
    cdef double[:,:] pure_coef_matrix
    cdef double[:,:] pure_coef_symbol_matrix
    cdef double[:,:] excess_coef_matrix
    cdef double[:,:] excess_coef_symbol_matrix
    cdef double[:,:] bm_coef_matrix
    cdef double[:,:] bm_coef_symbol_matrix
    cdef double[:,:] tc_coef_matrix
    cdef double[:,:] tc_coef_symbol_matrix
    cdef double ihj_magnetic_structure_factor
    cdef double afm_factor
    cdef int[:] disordered_sublattice_dof
    cdef double[:,:,::1] disordered_composition_matrices
    cdef double[:] disordered_site_ratios
    cdef double[:,:] disordered_pure_coef_matrix
    cdef double[:,:] disordered_pure_coef_symbol_matrix
    cdef double[:,:] disordered_excess_coef_matrix
    cdef double[:,:] disordered_excess_coef_symbol_matrix
    cdef double[:,:] disordered_bm_coef_matrix
    cdef double[:,:] disordered_bm_coef_symbol_matrix
    cdef double[:,:] disordered_tc_coef_matrix
    cdef double[:,:] disordered_tc_coef_symbol_matrix
    cdef double disordered_ihj_magnetic_structure_factor
    cdef double disordered_afm_factor
    cdef bint ordered

    cdef double _eval_rk_matrix(self, double[:,:] coef_mat, double[:,:] symbol_mat, double[:] dof,
                                double[:] eval_row, double[:] parameters) nogil
    cdef _eval_energy(self, double[:] out, double[:] dof, double[:] parameters, int out_idx, double sign)
    cpdef eval_energy(self, double[:] out, double[:,:] dof, double[:] parameters)