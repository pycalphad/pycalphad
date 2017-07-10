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
        nlp.addOption(b'derivative_test', b'first-order')
        nlp.addOption(b'check_derivatives_for_naninf', b'yes')
        nlp.addOption(b'print_level', 5)
        nlp.addOption(b'mu_strategy', b'adaptive')
        nlp.addOption(b'tol', 1e-6)
        nlp.addOption(b'acceptable_tol', 1.0)
        nlp.addOption(b'acceptable_constr_viol_tol', 1e-9)
        nlp.addOption(b'constr_viol_tol', 1e-12)
        # nlp.addOption(b'max_iter', 3000)
        x, info = nlp.solve(prob.x0)
        if info['status'] == -10:
            # Not enough degrees of freedom; nothing to do
            converged = True
        elif info['status'] < 0:
            if self.verbose:
                print('Calculation Failed: ', cur_conds, info['status_msg'])
                print(np.array(prob.num_vars), np.array(prob.num_constraints),
                      np.array(prob.xl), np.array(prob.xu), np.array(prob.cl), np.array(prob.cu))
            converged = False
        else:
            converged = True
        chemical_potentials = x[-prob.num_chempots:]

        return SolverResult(converged=converged, x=x, chemical_potentials=chemical_potentials)
