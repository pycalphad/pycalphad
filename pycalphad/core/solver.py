import ipopt
ipopt.setLoggingLevel(50)
import numpy as np
from collections import namedtuple

SolverResult = namedtuple('SolverResult', ['converged', 'x', 'chemical_potentials'])


class InteriorPointSolver(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def solve(self, prob):
        cur_conds = prob.conditions
        comps = prob.components
        nlp = ipopt.problem(
            n=prob.num_vars,
            m=prob.num_constraints,
            problem_obj=prob,
            lb=prob.xl,
            ub=prob.xu,
            cl=prob.cl,
            cu=prob.cu
        )
        # nlp.addOption(b'derivative_test', b'first-order')
        # nlp.addOption(b'check_derivatives_for_naninf', b'yes')
        MAX_SOLVE_DRIVING_FORCE = 1e-4
        nlp.addOption(b'print_level', 0)
        #nlp.addOption(b'mu_strategy', b'adaptive')
        #nlp.addOption(b'tol', 1e-2)
        #nlp.addOption(b'acceptable_tol', 1e-1)
        nlp.addOption(b'dual_inf_tol', MAX_SOLVE_DRIVING_FORCE)
        #nlp.addOption(b'compl_inf_tol', 1e-9)
        #nlp.addOption(b'acceptable_compl_inf_tol', 1e-9)
        nlp.addOption(b'bound_push', 1e-12)
        nlp.addOption(b'slack_bound_push', 1e-12)
        # This option improves convergence when using L-BFGS
        nlp.addOption(b'limited_memory_max_history', 100)
        nlp.addOption(b'acceptable_dual_inf_tol', MAX_SOLVE_DRIVING_FORCE)
        nlp.addOption(b'acceptable_constr_viol_tol', 1e-12)
        nlp.addOption(b'constr_viol_tol', 1e-12)
        nlp.addOption(b'bound_relax_factor', 1e-10)
        # nlp.addOption(b'max_iter', 3000)
        x, info = nlp.solve(prob.x0)
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
                print(np.array(prob.num_vars), np.array(prob.num_constraints),
                      np.array(prob.xl), np.array(prob.xu), np.array(prob.cl), np.array(prob.cu))
            converged = False
        else:
            converged = True
        print('Chemical Potentials', chemical_potentials)
        print(info['mult_x_L'])
        print(x)
        print('Status:', info['status'], info['status_msg'])
        return SolverResult(converged=converged, x=x, chemical_potentials=chemical_potentials)
