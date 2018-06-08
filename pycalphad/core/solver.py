import ipopt
ipopt.setLoggingLevel(50)
import numpy as np
from collections import namedtuple
from pycalphad.core.constants import MAX_SOLVE_DRIVING_FORCE, MAX_SOLVE_ITERATIONS
from pycalphad.variables import string_type

SolverResult = namedtuple('SolverResult', ['converged', 'x', 'chemical_potentials'])


class InteriorPointSolver(object):
    """
    Standard solver class that uses IPOPT.
    Should implement a ``solve`` method.

    Attributes
    ----------
    verbose : bool
        If True, will print solver diagonstics. Defaults to False.
    max_driving_force : float
        Maximum driving force allowed. Defaults to pycalphad.core.constants.MAX_SOLVE_DRIVING_FORCE.
        Used to tighten constraints, if necessary.

    Methods
    -------
    solve
        Solve a pycalphad.core.problem.Problem

    """

    def __init__(self, verbose=False, max_driving_force=MAX_SOLVE_DRIVING_FORCE, **ipopt_options):
        """
        Standard solver class that uses IPOPT.

        Parameters
        ----------
        verbose : bool
            If True, will print solver diagonstics. Defaults to False.
        max_driving_force : float
            Maximum driving force allowed
        ipopt_options : dict
            See https://www.coin-or.org/Ipopt/documentation/node40.html for all options

        """
        self.verbose = verbose
        self.max_driving_force = max_driving_force

        # set default options
        self.ipopt_options = {
            'max_iter': MAX_SOLVE_ITERATIONS,
            'print_level': 0,
            # This option improves convergence when using L-BFGS
            'limited_memory_max_history': 100,
            'tol': 1e-1,
            'constr_viol_tol': 1e-12
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
            if isinstance(value, string_type):
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
        length_scale = np.min(np.abs(prob.cl))
        length_scale = max(length_scale, 1e-9)
        x, info = nlp.solve(prob.x0)
        dual_inf = np.max(np.abs(info['mult_g']*info['g']))
        if dual_inf > self.max_driving_force:
            if self.verbose:
                print('Trying to improve poor solution')
            # Constraints are getting tiny; need to be strict about bounds
            if length_scale < 1e-6:
                nlp.addOption(b'compl_inf_tol', 1e-15)
                nlp.addOption(b'bound_relax_factor', 1e-12)
                # This option ensures any bounds failures will fail "loudly"
                # Otherwise we are liable to have subtle mass balance errors
                nlp.addOption(b'honor_original_bounds', b'no')
            else:
                nlp.addOption(b'dual_inf_tol', self.max_driving_force)
            accurate_x, accurate_info = nlp.solve(x)
            if accurate_info['status'] >= 0:
                x, info = accurate_x, accurate_info
        chemical_potentials = -np.array(info['mult_g'])[-len(set(comps) - {'VA'}):]
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
