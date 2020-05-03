import ipopt
ipopt.setLoggingLevel(50)
import numpy as np
from collections import namedtuple
from pycalphad.core.constants import MIN_SITE_FRACTION
from pycalphad.core.minimizer import find_solution

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
        compsets = prob.composition_sets
        state_variables = compsets[0].phase_record.state_variables
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
        prescribed_element_indices = np.array(prescribed_element_indices, dtype=np.int32)
        prescribed_elemental_amounts = np.array(prescribed_elemental_amounts)
        prescribed_system_amount = cur_conds.get('N', 1.0)
        free_chemical_potential_indices = np.array(sorted(set(range(num_components)) - set(prob.fixed_chempot_indices)), dtype=np.int32)
        fixed_chemical_potential_indices = np.array(prob.fixed_chempot_indices, dtype=np.int32)
        for fixed_chempot_index in fixed_chemical_potential_indices:
            el = prob.nonvacant_elements[fixed_chempot_index]
            chemical_potentials[fixed_chempot_index] = cur_conds.get('MU_' + str(el))
        free_stable_compset_indices = np.array(list(range(len(compsets))), dtype=np.int32)
        fixed_statevar_indices = []
        for statevar_idx, statevar in enumerate(state_variables):
            if str(statevar) in [str(k) for k in cur_conds.keys()]:
                fixed_statevar_indices.append(statevar_idx)
        free_statevar_indices = np.array(sorted(set(range(num_statevars)) - set(fixed_statevar_indices)), dtype=np.int32)
        fixed_statevar_indices = np.array(fixed_statevar_indices, dtype=np.int32)
        converged, x, chemical_potentials = find_solution(compsets, free_stable_compset_indices,
                  num_statevars, num_components, prescribed_system_amount,
                  chemical_potentials, free_chemical_potential_indices, fixed_chemical_potential_indices,
                  prescribed_element_indices, prescribed_elemental_amounts,
                  free_statevar_indices, fixed_statevar_indices)

        if self.verbose:
            print('Chemical Potentials', chemical_potentials)
            print(x)
        return SolverResult(converged=converged, x=x, chemical_potentials=chemical_potentials)
