import cyipopt
cyipopt.set_logging_level(50)
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
        problem : cyipopt.Problem
            A problem object that will be solved

        Notes
        -----
        Strings are encoded to byte strings.
        """
        for option, value in self.ipopt_options.items():
            if isinstance(value, str):
                problem.add_option(option.encode(), value.encode())
            else:
                problem.add_option(option.encode(), value)


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
        nlp = cyipopt.Problem(
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
            nlp.add_option(b'hessian_approximation', b'limited-memory')
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
            nlp.add_option(b'compl_inf_tol', 1e-3 * float(length_scale))
            nlp.add_option(b'bound_relax_factor', MIN_SITE_FRACTION)
            # This option ensures any bounds failures will fail "loudly"
            # Otherwise we are liable to have subtle mass balance errors
            nlp.add_option(b'honor_original_bounds', b'no')
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
