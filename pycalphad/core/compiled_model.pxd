from pycalphad.core.phase_rec cimport func_t, func_novec_t

cdef public class CompiledModel(object)[type CompiledModelType, object CompiledModelObject]:
    cdef public object constituents
    cdef public object variables
    cdef public object components
    cdef readonly unicode phase_name
    cdef public int[::1] sublattice_dof
    cdef public int phase_dof
    cdef public double[:,:,::1] composition_matrices
    cdef public double[::1] site_ratios
    cdef public int vacancy_index
    cdef public double[:,:] pure_coef_matrix
    cdef public double[:,:] pure_coef_symbol_matrix
    cdef public double[:,:] excess_coef_matrix
    cdef public double[:,:] excess_coef_symbol_matrix
    cdef public double[:,:] bm_coef_matrix
    cdef public double[:,:] bm_coef_symbol_matrix
    cdef public double[:,:] tc_coef_matrix
    cdef public double[:,:] tc_coef_symbol_matrix
    cdef public double ihj_magnetic_structure_factor
    cdef public double afm_factor
    cdef public int[:] disordered_sublattice_dof
    cdef public int disordered_phase_dof
    cdef public double[:,:,::1] disordered_composition_matrices
    cdef public double[:] disordered_site_ratios
    cdef public double[:,:] disordered_pure_coef_matrix
    cdef public double[:,:] disordered_pure_coef_symbol_matrix
    cdef public double[:,:] disordered_excess_coef_matrix
    cdef public double[:,:] disordered_excess_coef_symbol_matrix
    cdef public double[:,:] disordered_bm_coef_matrix
    cdef public double[:,:] disordered_bm_coef_symbol_matrix
    cdef public double[:,:] disordered_tc_coef_matrix
    cdef public double[:,:] disordered_tc_coef_symbol_matrix
    cdef public double disordered_ihj_magnetic_structure_factor
    cdef public double disordered_afm_factor
    cdef public bint ordered
    cdef public bint _debug
    cdef func_t* _debugobj
    cdef func_novec_t* _debuggrad
    cdef func_novec_t* _debughess

    cdef double _eval_rk_matrix(self, double[:,:] coef_mat, double[:,:] symbol_mat,
                                double *eval_row, double[:] parameters) nogil
    cdef void _eval_rk_matrix_gradient(self, double *out, double[:,:] coef_mat, double[:,:] symbol_mat,
                                           double *eval_row, double[:] parameters) nogil
    cdef void _compute_disordered_dof(self, double *disordered_dof, double *dof, size_t num_pts) nogil
    cdef void _compute_ordered_dof(self, double *ordered_dof, double *disordered_dof, size_t num_pts) nogil
    cdef void _eval_energy(self, double *out, double *dof, double[:] parameters, double sign, size_t num_pts) nogil
    cdef void _eval_disordered_energy(self, double *out, double *dof, double[:] parameters, double sign) nogil
    cdef void eval_energy(self, double *out, double *dof, double[:] parameters, size_t num_pts) nogil
    cdef void _eval_energy_gradient(self, double *out_grad, double *dof, double[:] parameters, double sign) nogil
    cdef void eval_energy_gradient(self, double *out, double *dof, double[:] parameters) nogil
    cdef _debug_energy(self, double[::1] debugout, double[:,::1] dof, double[::1] parameters)
    cdef _debug_energy_gradient(self, double[::1] debugout, double[::1] dof, double[::1] parameters)
    cdef void eval_energy_hessian(self, double[:, ::1] out, double[:] dof, double[:] parameters) nogil
