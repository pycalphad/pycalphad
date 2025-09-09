# distutils: language = c++
cpdef double hyperplane(double[:,::1] compositions,
                        double[::1] energies,
                        double[::1] chemical_potentials,
                        size_t[::1] fixed_chempot_indices,
                        double[:, ::1] fixed_lincomb_molefrac_coefs,
                        double[::1] fixed_lincomb_molefrac_rhs,
                        double[::1] result_fractions,
                        int[::1] result_simplex) except *