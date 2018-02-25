import ipopt
ipopt.setLoggingLevel(50)
import numpy as np
from collections import namedtuple
from pycalphad.core.constants import MAX_SOLVE_DRIVING_FORCE

SolverResult = namedtuple('SolverResult', ['converged', 'x', 'chemical_potentials'])


class InteriorPointSolver(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def solve(self, prob):
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
        length_scale = np.min(np.abs(prob.cl))
        length_scale = max(length_scale, 1e-9)
        nlp.addOption(b'print_level', 0)
        if not self.verbose:
            # suppress the "This program contains Ipopt" banner
            nlp.addOption(b'sb', b'yes')
        nlp.addOption(b'tol', 1e-1)
        nlp.addOption(b'constr_viol_tol', 1e-12)
        # This option improves convergence when using L-BFGS
        nlp.addOption(b'limited_memory_max_history', 100)
        nlp.addOption(b'max_iter', 200)
        x, info = nlp.solve(prob.x0)
        dual_inf = np.max(np.abs(info['mult_g']*info['g']))
        if dual_inf > MAX_SOLVE_DRIVING_FORCE:
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
                nlp.addOption(b'dual_inf_tol', MAX_SOLVE_DRIVING_FORCE)
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
