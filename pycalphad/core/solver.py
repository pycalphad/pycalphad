import ipopt
ipopt.setLoggingLevel(50)
import numpy as np
from collections import namedtuple
from pycalphad.core.constants import MIN_SITE_FRACTION

SolverResult = namedtuple('SolverResult', ['converged', 'x', 'chemical_potentials'])

class SolverBase(object):
    """"Base class for solvers."""
    ignore_convergence = False
    def solve(self, prob):
        """
        *Implement this method.*
        Solve a non-linear problem

        Parameters
        ----------
        prob : pycalphad.core.problem.Problem

        Returns
        -------
        pycalphad.core.solver.SolverResult
        """
        raise NotImplementedError("A subclass of Solver must be implemented.")


class InteriorPointSolver(SolverBase):
    """
    Standard solver class that uses IPOPT.

    Attributes
    ----------
    verbose : bool
        If True, will print solver diagonstics. Defaults to False.
    infeasibility_threshold : float
        Dual infeasibility threshold used to tighten constraints and
        attempt a second solve, if necessary. Defaults to 1e-4.
    ipopt_options : dict
        Dictionary of options to pass to IPOPT.

    Methods
    -------
    solve
        Solve a pycalphad.core.problem.Problem
    apply_options
        Encodes ipopt_options and applies them to problem

    """

    def __init__(self, verbose=False, infeasibility_threshold=1e-4, **ipopt_options):
        """
        Standard solver class that uses IPOPT.

        Parameters
        ----------
        verbose : bool
            If True, will print solver diagonstics. Defaults to False.
        infeasibility_threshold : float
            Dual infeasibility threshold used to tighten constraints and
            attempt a second solve, if necessary. Defaults to 1e-4.
        ipopt_options : dict
            See https://www.coin-or.org/Ipopt/documentation/node40.html for all options

        """
        self.verbose = verbose
        self.infeasibility_threshold = infeasibility_threshold

        # set default options
        self.ipopt_options = {
            'max_iter': 200,
            'print_level': 0,
            'tol': 1e-1,
            'constr_viol_tol': 1e-5,
            'nlp_scaling_method': 'none',
            'hessian_approximation': 'exact'
        }
        if not self.verbose:
            # suppress the "This program contains Ipopt" banner
            self.ipopt_options['sb'] = ipopt_options.pop('sb', 'yes')

        # update the default options with the passed options
        self.ipopt_options.update(ipopt_options)


    def apply_options(self, problem):
        """
        Apply global options to the solver

        Parameters
        ----------
        problem : ipopt.problem
            A problem object that will be solved

        Notes
        -----
        Strings are encoded to byte strings.
        """
        for option, value in self.ipopt_options.items():
            if isinstance(value, str):
                problem.addOption(option.encode(), value.encode())
            else:
                problem.addOption(option.encode(), value)


    def solve(self, prob):
        """
        Solve a non-linear problem

        Parameters
        ----------
        prob : pycalphad.core.problem.Problem

        Returns
        -------
        SolverResult

        """
        cur_conds = prob.conditions
        comps = prob.pure_elements
        nlp = ipopt.problem(
            n=prob.num_vars,
            m=prob.num_constraints,
            problem_obj=prob,
            lb=prob.xl,
            ub=prob.xu,
            cl=prob.cl,
            cu=prob.cu
        )
        self.apply_options(nlp)
        # XXX: Hack until exact chemical potential Hessians are implemented
        if len(prob.fixed_chempot_indices) > 0:
            nlp.addOption(b'hessian_approximation', b'limited-memory')
            if self.verbose:
                print('Turning off exact Hessians due to advanced condition')
        # Note: Using the ipopt derivative checker can be tricky at the edges of composition space
        # It will not give valid results for the finite difference approximation
        x, info = nlp.solve(prob.x0)
        length_scale = max(np.min(np.abs(x)), 1e-9)
        if length_scale < 1e-2:
            if self.verbose:
                print('Trying to improve poor solution')
            # Constraints are getting tiny; need to be strict about bounds
            nlp.addOption(b'compl_inf_tol', 1e-3 * float(length_scale))
            nlp.addOption(b'bound_relax_factor', MIN_SITE_FRACTION)
            # This option ensures any bounds failures will fail "loudly"
            # Otherwise we are liable to have subtle mass balance errors
            nlp.addOption(b'honor_original_bounds', b'no')
            accurate_x, accurate_info = nlp.solve(x)
            if accurate_info['status'] >= 0:
                x, info = accurate_x, accurate_info
        chemical_potentials = prob.chemical_potentials(x)
        if info['status'] == -10:
            # Not enough degrees of freedom; nothing to do
            if len(prob.composition_sets) == 1:
                converged = True
                chemical_potentials[:] = prob.composition_sets[0].energy
            else:
                converged = False
        elif info['status'] < 0:
            if self.verbose:
                print('Calculation Failed: ', cur_conds, info['status_msg'])
            converged = False
        else:
            converged = True
        if self.verbose:
            print('Chemical Potentials', chemical_potentials)
            print(info['mult_x_L'])
            print(x)
            print('Status:', info['status'], info['status_msg'])
        return SolverResult(converged=converged, x=x, chemical_potentials=chemical_potentials)


class SundmanSolver(SolverBase):
    def __init__(self, verbose=False, **options):
        self.verbose = verbose

    def solve(self, prob):
        """
        Solve a non-linear problem

        Parameters
        ----------
        prob : pycalphad.core.problem.Problem

        Returns
        -------
        SolverResult

        """
        cur_conds = prob.conditions
        print(cur_conds)
        compsets = prob.composition_sets
        state_variables = compsets[0].phase_record.state_variables
        print('state_variables', state_variables)
        num_statevars = len(state_variables)
        num_components = len(prob.nonvacant_elements)
        chemical_potentials = prob.chemical_potentials(prob.x0)
        prescribed_elemental_amounts = []
        prescribed_element_indices = []
        for cond, value in cur_conds.items():
            if str(cond).startswith('X_'):
                el = str(cond)[2:]
                el_idx = list(prob.nonvacant_elements).index(el)
                prescribed_elemental_amounts.append(float(value))
                prescribed_element_indices.append(el_idx)
        prescribed_system_amount = cur_conds.get('N', 1.0)

        phase_amt = np.array([compset.NP for compset in compsets])

        dof = [np.array(compset.dof) for compset in compsets]
        print('dof', dof)

        free_chemical_potential_indices = np.array(sorted(set(range(num_components)) - set(prob.fixed_chempot_indices)))
        fixed_chemical_potential_indices = np.array(prob.fixed_chempot_indices)
        for fixed_chempot_index in fixed_chemical_potential_indices:
            el = prob.nonvacant_elements[fixed_chempot_index]
            chemical_potentials[fixed_chempot_index] = cur_conds.get('MU_' + str(el))
        free_stable_compset_indices = np.array(list(range(len(compsets))))
        fixed_statevar_indices = []
        for statevar_idx, statevar in enumerate(state_variables):
            if str(statevar) in [str(k) for k in cur_conds.keys()]:
                fixed_statevar_indices.append(statevar_idx)
        free_statevar_indices = np.array(sorted(set(range(num_statevars)) - set(fixed_statevar_indices)))
        print('free_chemical_potential_indices', free_chemical_potential_indices)
        print('fixed_chemical_potential_indices', fixed_chemical_potential_indices)
        print('free_stable_compset_indices', free_stable_compset_indices)
        print('fixed_statevar_indices', fixed_statevar_indices)
        print('free_statevar_indices', free_statevar_indices)
        print('prescribed_elemental_amounts', prescribed_elemental_amounts)
        print('prescribed_element_indices', prescribed_element_indices)
        print('prescribed_system_amount', prescribed_system_amount)
        print('phase_amt', phase_amt)
        delta_statevars = np.zeros(num_statevars)
        iterations_since_phase_change = 0
        freeze_phase_internal_dof = False
        for iteration in range(100):
            current_elemental_amounts = np.zeros_like(chemical_potentials)
            all_phase_energies = np.zeros((len(compsets), 1))
            all_phase_amounts = np.zeros((len(compsets), len(chemical_potentials)))
            largest_statevar_change = 0
            largest_internal_dof_change = 0
            largest_phase_amt_change = 0
            # FIRST STEP: Update phase internal degrees of freedom
            for idx, compset in enumerate(compsets):
                # TODO: Use better dof storage
                x = dof[idx]
                # Compute phase matrix (LHS of Eq. 41, Sundman 2015)
                phase_matrix = np.zeros((compset.phase_record.phase_dof + compset.phase_record.num_internal_cons,
                                         compset.phase_record.phase_dof + compset.phase_record.num_internal_cons))
                hess_tmp = np.zeros((num_statevars + compset.phase_record.phase_dof,
                                     num_statevars + compset.phase_record.phase_dof))
                cons_jac_tmp = np.zeros((compset.phase_record.num_internal_cons,
                                         num_statevars + compset.phase_record.phase_dof))
                compset.phase_record.hess(hess_tmp, x)
                phase_matrix[:compset.phase_record.phase_dof, :compset.phase_record.phase_dof] = hess_tmp[
                                                                                                 num_statevars:,
                                                                                                 num_statevars:]
                compset.phase_record.internal_cons_jac(cons_jac_tmp, x)
                phase_matrix[compset.phase_record.phase_dof:, :compset.phase_record.phase_dof] = cons_jac_tmp[:,
                                                                                                 num_statevars:]
                phase_matrix[:compset.phase_record.phase_dof, compset.phase_record.phase_dof:] = cons_jac_tmp[:,
                                                                                                 num_statevars:].T

                # Compute right-hand side of Eq. 41, Sundman 2015
                rhs = np.zeros(compset.phase_record.phase_dof + compset.phase_record.num_internal_cons)
                grad_tmp = np.zeros(num_statevars + compset.phase_record.phase_dof)
                compset.phase_record.grad(grad_tmp, x)
                rhs[:compset.phase_record.phase_dof] = -grad_tmp[num_statevars:]
                rhs[:compset.phase_record.phase_dof] -= np.dot(hess_tmp[num_statevars:, :num_statevars],
                                                               delta_statevars)
                mass_jac_tmp = np.zeros((num_components, num_statevars + compset.phase_record.phase_dof))
                for comp_idx in range(num_components):
                    compset.phase_record.mass_grad(mass_jac_tmp[comp_idx, :], x, comp_idx)
                rhs[:compset.phase_record.phase_dof] += mass_jac_tmp.T[num_statevars:].dot(chemical_potentials)
                if not freeze_phase_internal_dof:
                    soln = np.linalg.solve(phase_matrix, rhs)
                    delta_y = soln[:compset.phase_record.phase_dof]
                    largest_internal_dof_change = max(largest_internal_dof_change, np.max(np.abs(delta_y)))
                    old_y = np.array(x[num_statevars:])
                    new_y = old_y + delta_y
                    new_y[new_y < 1e-15] = 1e-15
                    new_y[new_y > 1] = 1
                    x[num_statevars:] = new_y

                masses_tmp = np.zeros((num_components, 1))
                for comp_idx in range(num_components):
                    compset.phase_record.mass_obj(masses_tmp[comp_idx, :], x, comp_idx)
                    all_phase_amounts[idx, comp_idx] = masses_tmp[comp_idx, 0]
                    if phase_amt[idx] > 0:
                        current_elemental_amounts[comp_idx] += phase_amt[idx] * masses_tmp[comp_idx, 0]
                compset.phase_record.obj(all_phase_energies[idx, :], x)
                # print(compset.phase_record.phase_name, idx, new_y)
            # SECOND STEP: Update potentials and phase amounts, according to conditions
            num_stable_phases = free_stable_compset_indices.shape[0]
            num_fixed_components = len(prescribed_elemental_amounts)
            num_free_variables = free_chemical_potential_indices.shape[0] + num_stable_phases + \
                                 free_statevar_indices.shape[0]
            equilibrium_matrix = np.zeros((num_stable_phases + num_fixed_components + 1, num_free_variables))
            equilibrium_rhs = np.zeros(num_stable_phases + num_fixed_components + 1)
            if (num_stable_phases + num_fixed_components + 1) != num_free_variables:
                raise ValueError('Conditions do not obey Gibbs Phase Rule')
            for stable_idx in range(free_stable_compset_indices.shape[0]):
                idx = free_stable_compset_indices[stable_idx]
                compset = compsets[idx]
                # TODO: Use better dof storage
                # Calculate key phase quantities starting here
                x = dof[idx]
                # print('x', x)
                energy_tmp = np.zeros((1, 1))
                compset.phase_record.obj(energy_tmp[:, 0], x)
                masses_tmp = np.zeros((num_components, 1))
                mass_jac_tmp = np.zeros((num_components, num_statevars + compset.phase_record.phase_dof))
                for comp_idx in range(num_components):
                    compset.phase_record.mass_grad(mass_jac_tmp[comp_idx, :], x, comp_idx)
                    compset.phase_record.mass_obj(masses_tmp[comp_idx, :], x, comp_idx)
                # Compute phase matrix (LHS of Eq. 41, Sundman 2015)
                phase_matrix = np.zeros((compset.phase_record.phase_dof + compset.phase_record.num_internal_cons,
                                         compset.phase_record.phase_dof + compset.phase_record.num_internal_cons))
                hess_tmp = np.zeros((num_statevars + compset.phase_record.phase_dof,
                                     num_statevars + compset.phase_record.phase_dof))
                cons_jac_tmp = np.zeros((compset.phase_record.num_internal_cons,
                                         num_statevars + compset.phase_record.phase_dof))
                compset.phase_record.hess(hess_tmp, x)
                grad_tmp = np.zeros(num_statevars + compset.phase_record.phase_dof)
                compset.phase_record.grad(grad_tmp, x)
                phase_matrix[:compset.phase_record.phase_dof, :compset.phase_record.phase_dof] = hess_tmp[
                                                                                                 num_statevars:,
                                                                                                 num_statevars:]
                compset.phase_record.internal_cons_jac(cons_jac_tmp, x)
                phase_matrix[compset.phase_record.phase_dof:, :compset.phase_record.phase_dof] = cons_jac_tmp[:,
                                                                                                 num_statevars:]
                phase_matrix[:compset.phase_record.phase_dof, compset.phase_record.phase_dof:] = cons_jac_tmp[:,
                                                                                                 num_statevars:].T
                e_matrix = np.linalg.inv(phase_matrix)[:compset.phase_record.phase_dof, :compset.phase_record.phase_dof]
                # Eq. 44
                c_G = -np.dot(e_matrix, grad_tmp[num_statevars:])
                c_statevars = -np.dot(e_matrix, hess_tmp[num_statevars:, :num_statevars])
                c_component = np.dot(mass_jac_tmp[:, num_statevars:], e_matrix)
                # Calculations of key quantities complete

                # KEY STEPS for filling equilibrium matrix
                # 1. Contribute to the row corresponding to this composition set
                # 1a. Loop through potential conditions to fill out each column
                # 2. Contribute to the rows of all fixed components
                # 2a. Loop through potential conditions to fill out each column
                # 3. Contribute to RHS of each component row
                # 4. Add energies to RHS of each stable composition set
                # 5. Subtract contribution from RHS due to any fixed chemical potentials
                # 6. Subtract fixed chemical potentials from each fixed component RHS

                # 1a. This phase row: free chemical potentials
                free_variable_column_offset = 0
                for i in range(free_chemical_potential_indices.shape[0]):
                    chempot_idx = free_chemical_potential_indices[i]
                    equilibrium_matrix[stable_idx, free_variable_column_offset + i] = masses_tmp[chempot_idx, 0]
                free_variable_column_offset += free_chemical_potential_indices.shape[0]
                # 1a. This phase row: free stable composition sets = zero contribution
                free_variable_column_offset += free_stable_compset_indices.shape[0]
                # 1a. This phase row: free state variables
                for i in range(free_statevar_indices.shape[0]):
                    statevar_idx = free_statevar_indices[i]
                    equilibrium_matrix[stable_idx, free_variable_column_offset + i] = -grad_tmp[statevar_idx]

                # 2. Contribute to the row of all fixed components
                component_row_offset = num_stable_phases
                for fixed_component_idx in range(num_fixed_components):
                    component_idx = prescribed_element_indices[fixed_component_idx]
                    free_variable_column_offset = 0
                    # 2a. This component row: free chemical potentials
                    for i in range(free_chemical_potential_indices.shape[0]):
                        chempot_idx = free_chemical_potential_indices[i]
                        equilibrium_matrix[component_row_offset + fixed_component_idx, free_variable_column_offset + i] += \
                            phase_amt[idx] * np.dot(mass_jac_tmp[component_idx, num_statevars:],
                                                    c_component[chempot_idx, :])
                    free_variable_column_offset += free_chemical_potential_indices.shape[0]
                    # 2a. This component row: free stable composition sets
                    for i in range(free_stable_compset_indices.shape[0]):
                        compset_idx = free_stable_compset_indices[i]
                        # Only fill this out if the current idx is equal to a free composition set
                        if compset_idx == idx:
                            equilibrium_matrix[component_row_offset + fixed_component_idx, free_variable_column_offset + i] = \
                            masses_tmp[component_idx, 0]
                    free_variable_column_offset += free_stable_compset_indices.shape[0]
                    # 2a. This component row: free state variables
                    for i in range(free_statevar_indices.shape[0]):
                        statevar_idx = free_statevar_indices[i]
                        equilibrium_matrix[component_row_offset + fixed_component_idx, free_variable_column_offset + i] += \
                            phase_amt[idx] * np.dot(mass_jac_tmp[component_idx, num_statevars:],
                                                    c_statevars[:, statevar_idx])
                    # 3.
                    equilibrium_rhs[component_row_offset + fixed_component_idx] += -phase_amt[idx] * np.dot(
                        mass_jac_tmp[component_idx, num_statevars:], c_G)

                system_amount_index = component_row_offset + num_fixed_components
                # 2X. Also handle the N=1 row
                for component_idx in range(num_components):
                    free_variable_column_offset = 0
                    # 2a. This component row: free chemical potentials
                    for i in range(free_chemical_potential_indices.shape[0]):
                        chempot_idx = free_chemical_potential_indices[i]
                        #equilibrium_matrix[system_amount_index, free_variable_column_offset + i] += \
                        #    phase_amt[idx] * np.dot(mass_jac_tmp[component_idx, num_statevars:],
                        #                            c_component[chempot_idx, :])
                    free_variable_column_offset += free_chemical_potential_indices.shape[0]
                    # 2a. This component row: free stable composition sets
                    for i in range(free_stable_compset_indices.shape[0]):
                        compset_idx = free_stable_compset_indices[i]
                        # Only fill this out if the current idx is equal to a free composition set
                        if compset_idx == idx:
                            equilibrium_matrix[system_amount_index, free_variable_column_offset + i] = 1
                    free_variable_column_offset += free_stable_compset_indices.shape[0]
                    # 2a. This component row: free state variables
                    for i in range(free_statevar_indices.shape[0]):
                        statevar_idx = free_statevar_indices[i]
                        equilibrium_matrix[system_amount_index, free_variable_column_offset + i] += \
                            phase_amt[idx] * np.dot(mass_jac_tmp[component_idx, num_statevars:],
                                                    c_statevars[:, statevar_idx])
                    # 3.
                    equilibrium_rhs[system_amount_index] += -phase_amt[idx] * np.dot(
                        mass_jac_tmp[component_idx, num_statevars:], c_G)
                # 4.
                equilibrium_rhs[idx] = energy_tmp[0, 0]
                # 5. Subtract fixed chemical potentials from each phase RHS
                for i in range(fixed_chemical_potential_indices.shape[0]):
                    chempot_idx = fixed_chemical_potential_indices[i]
                    equilibrium_rhs[idx] -= masses_tmp[chempot_idx, :] * chemical_potentials[chempot_idx]
                    # 6. Subtract fixed chemical potentials from each fixed component RHS
                    for fixed_component_idx in range(num_fixed_components):
                        component_idx = prescribed_element_indices[fixed_component_idx]
                        equilibrium_rhs[component_row_offset + fixed_component_idx] -= phase_amt[idx] * chemical_potentials[
                            chempot_idx] * np.dot(mass_jac_tmp[component_idx, num_statevars:],
                                                  c_component[chempot_idx, :])
                    #for component_idx in range(num_components):
                    #    equilibrium_rhs[system_amount_index] -= phase_amt[idx] * chemical_potentials[
                    #        chempot_idx] * np.dot(mass_jac_tmp[component_idx, num_statevars:],
                    #                             c_component[chempot_idx, :])

            # Add mass residual to fixed component row RHS, plus N=1 row
            mass_residual = 0.0
            component_row_offset = num_stable_phases
            system_amount_index = component_row_offset + num_fixed_components
            current_system_amount = float(phase_amt.sum())
            print('current_system_amount', current_system_amount)
            print('prescribed_system_amount', prescribed_system_amount)
            print('current_elemental_amounts', current_elemental_amounts)
            for fixed_component_idx in range(num_fixed_components):
                component_idx = prescribed_element_indices[fixed_component_idx]
                mass_residual += abs(current_elemental_amounts[component_idx] - prescribed_elemental_amounts[fixed_component_idx])
                equilibrium_rhs[component_row_offset + fixed_component_idx] -= current_elemental_amounts[component_idx] - prescribed_elemental_amounts[fixed_component_idx]
            mass_residual += abs(current_system_amount - prescribed_system_amount)
            equilibrium_rhs[system_amount_index] -= current_system_amount - prescribed_system_amount
            equilibrium_soln = np.linalg.lstsq(equilibrium_matrix, equilibrium_rhs)[0]
            soln_index_offset = 0
            for i in range(free_chemical_potential_indices.shape[0]):
                chempot_idx = free_chemical_potential_indices[i]
                chempot_change = equilibrium_soln[soln_index_offset + i] - chemical_potentials[chempot_idx]
                percent_chempot_change = abs(chempot_change / chemical_potentials[chempot_idx])
                chemical_potentials[chempot_idx] = equilibrium_soln[soln_index_offset + i]
                largest_statevar_change = max(largest_statevar_change, percent_chempot_change)
            soln_index_offset += free_chemical_potential_indices.shape[0]
            for i in range(free_stable_compset_indices.shape[0]):
                compset_idx = free_stable_compset_indices[i]
                phase_amt_change = float(phase_amt[compset_idx])
                phase_amt[compset_idx] += equilibrium_soln[soln_index_offset + i]
                phase_amt[compset_idx] = np.minimum(1.0, phase_amt[compset_idx])
                phase_amt[compset_idx] = np.maximum(0.0, phase_amt[compset_idx])
                phase_amt_change = phase_amt[compset_idx] - phase_amt_change
                largest_phase_amt_change = max(largest_phase_amt_change, phase_amt_change)
                print('Updating phase_amt for compset ', compset_idx, ' by ', equilibrium_soln[soln_index_offset + i])

            soln_index_offset += free_stable_compset_indices.shape[0]
            delta_statevars[:] = 0
            for i in range(free_statevar_indices.shape[0]):
                statevar_idx = free_statevar_indices[i]
                delta_statevars[statevar_idx] = equilibrium_soln[soln_index_offset + i]
            percent_statevar_changes = np.abs(delta_statevars / dof[0][:num_statevars])
            percent_statevar_changes[np.isnan(percent_statevar_changes)] = 0
            largest_statevar_change = max(largest_statevar_change, np.max(percent_statevar_changes))
            for idx in range(len(dof)):
                dof[idx][:num_statevars] += delta_statevars

            # Wait for mass balance to be satisfied before changing phases
            # Phases that "want" to be removed will keep having their phase_amt set to zero, so mass balance is unaffected
            #
            system_is_feasible = mass_residual < 1e-10
            print('system_is_feasible', system_is_feasible)
            print('largest_internal_dof_change', largest_internal_dof_change)
            print('largest_phase_amt_change', largest_phase_amt_change)
            print('largest_statevar_change', largest_statevar_change)
            if system_is_feasible:
                freeze_phase_internal_dof = False
                print('freeze_phase_internal_dof = False', freeze_phase_internal_dof)
                iterations_since_phase_change = 0
                free_stable_compset_indices = np.nonzero(phase_amt > MIN_SITE_FRACTION)[0]
                # Check driving forces for metastable phases
                for idx in range(len(compsets)):
                    all_phase_energies[idx, 0] -= np.dot(chemical_potentials, all_phase_amounts[idx, :])
                print('Driving Forces: ', all_phase_energies[:, 0])
                compsets_to_add = set(np.nonzero(all_phase_energies[:, 0] > -1e-5)[0])
                current_free_stable_compset_indices = free_stable_compset_indices
                new_free_stable_compset_indices = np.array(sorted(set(free_stable_compset_indices) | compsets_to_add))
                if len(set(current_free_stable_compset_indices) - set(new_free_stable_compset_indices)) != 0:
                    freeze_phase_internal_dof = True
                    print('freeze_phase_internal_dof = True', freeze_phase_internal_dof)
                else:
                    # feasible system, and no phases to add or remove
                    if (largest_internal_dof_change < 1e-13) and (largest_phase_amt_change < 1e-10) and \
                        (largest_statevar_change < 1e-3):
                        converged = True
                        print('CONVERGED')
                        break
                free_stable_compset_indices = new_free_stable_compset_indices
            else:
                iterations_since_phase_change += 1
            print('free_stable_compset_indices', free_stable_compset_indices)

            print('NP', phase_amt, 'MU', chemical_potentials, 'statevars', dof[0][:num_statevars])

        x = dof[0]
        for cs_dof in dof[1:]:
            x = np.r_[x, cs_dof[num_statevars:]]
        x = np.r_[x, phase_amt]
        print('Result x', x)

        # TODO: Do not force convergence
        converged = True
        if self.verbose:
            print('Chemical Potentials', chemical_potentials)
            print(x)
        return SolverResult(converged=converged, x=x, chemical_potentials=chemical_potentials)
