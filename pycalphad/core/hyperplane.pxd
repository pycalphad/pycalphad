# distutils: language = c++
cpdef double hyperplane(double[:,::1] compositions,
                        double[::1] energies,
                        double[::1] composition,
                        double[::1] chemical_potentials,
                        double total_moles,
                        size_t[::1] fixed_chempot_indices,
                        size_t[::1] fixed_comp_indices,
                        double[::1] result_fractions,
                        int[::1] result_simplex) nogil except *
