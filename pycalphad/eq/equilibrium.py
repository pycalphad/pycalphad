"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""

import pycalphad.variables as v
from pycalphad.eq.utils import make_callable, generate_dof
from pycalphad.eq.utils import check_degenerate_phases
from pycalphad.constraints import sitefrac_cons, sitefrac_jac
from pycalphad.constraints import molefrac_ast
from pycalphad import Model
from pycalphad.eq.energy_surf import energy_surf
from pycalphad.eq.simplex import Simplex
import pandas as pd
import numpy as np
from sympy import Add
import scipy.spatial
import scipy.optimize
from collections import Counter

try:
    set
except NameError:
    from sets import Set as set #pylint: disable=W0622

class Equilibrium(object):
    """
    Calculate the equilibrium state of a system containing the specified
    components and phases, under the specified conditions.
    Model parameters are taken from 'db' and any state variables (T, P, etc.)
    can be specified as keyword arguments.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names (case-sensitive) of components to consider in the calculation.
    phases : list
        Names (case-sensitive) of phases to consider in the calculation.
    conditions : dict
        StateVariables and their corresponding value.

    Returns
    -------
    Structured equilibrium calculation.

    Examples
    --------
    None yet.
    """
    def __init__(self, dbf, comps, phases, conditions, **kwargs):
        self.conditions = conditions
        self.components = set(comps)
        self.phases = dict()
        self.statevars = dict()
        self.data = pd.DataFrame()

        self._phases = dict([[name, dbf.phases[name]] for name in phases])
        self._phase_callables = dict()
        self._gradient_callables = dict()
        self._molefrac_callables = dict()
        self._molefrac_jac_callables = dict()
        self._variables = dict()
        self._sublattice_dof = dict()
        self.statevars = dict()
        for key in ['T', 'P']:
            try:
                self.statevars[v.StateVariable(key)] = kwargs[key]
            except KeyError:
                pass

        self._build_objective_functions(dbf)

        self.data = energy_surf(dbf, comps, phases, **kwargs)

        # self.data now contains energy surface information for the system
        # find simplex for a starting point; refine with optimization
        estimates = self.get_starting_simplex()
        print(estimates)
        self.result = self.minimize(estimates[0], estimates[1])

    def __repr__(self):
        estimate = self.get_starting_simplex()
        variables = []
        for name in estimate[0]['Phase'].values:
            variables.append(name+'_FRAC')
            variables.extend(self._variables[name])
        return str(self.result)+'\n'+str(dict(zip(variables, self.result.x)))

    def get_starting_simplex(self):
        """
        Calculate convex hull and find a suitable starting point.
        Returns iterable: first is a DataFrame of the phase compositions
                          second is an estimate of the phase fractions
        """
        # determine column indices for independent degrees of freedom
        independent_dof = []
        independent_dof_values = []
        for cond, value in self.conditions.items():
            if not isinstance(cond, v.Composition):
                continue
            # ignore phase-specific composition conditions
            if cond.phase_name is not None:
                continue
            independent_dof.append('X(' + cond.species + ')')
            independent_dof_values.append(value)
        if len(independent_dof) > 1:
            independent_dof.pop()
            independent_dof_values.pop()

        independent_dof.append('GM')
        # calculate the convex hull for the independent d.o.f
        # TODO: Apply activity conditions here
        hull = scipy.spatial.ConvexHull(
            self.data[independent_dof].values, qhull_options='QJ'
        )
        # locate the simplex closest to our desired condition
        candidate_simplex = None
        phase_fracs = None
        for equ, simp in zip(hull.equations, hull.simplices):
            if equ[-2] < 0:
                new_simp = None
                try:
                    new_simp = Simplex(hull.points[simp, :-1])
                except np.linalg.LinAlgError:
                    print('skipping '+str(simp))
                    continue
                if new_simp.in_simplex(independent_dof_values):
                    candidate_simplex = simp
                    phase_fracs = new_simp.bary_coords(independent_dof_values)
                    break
        phase_compositions = self.data.iloc[candidate_simplex]
        independent_indices = check_degenerate_phases(phase_compositions)
        # renormalize phase fractions to 1 after eliminating redundant phases
        phase_fracs = phase_fracs[independent_indices]
        phase_fracs /= np.sum(phase_fracs)
        return [phase_compositions.iloc[independent_indices], phase_fracs]

    def minimize(self, simplex, phase_fractions=None):
        """
        Accept a list of simplex vertices and return the values of the
        variables that minimize the energy under the constraints.
        """
        # Generate phase fraction variables
        # Track the multiplicity of phases with a Counter object
        composition_sets = Counter()
        all_variables = []
        # starting point
        x_0 = []
        # a list of tuples for where each phase's variable indices
        # start and end
        index_ranges = []
        #print(list(enumerate(simplex.iterrows())))
        #print((simplex.iterrows()))
        #print('END')
        for m_idx, vertex in enumerate(simplex.iterrows()):
            vertex = vertex[1]
            # increase multiplicity by one
            composition_sets[vertex['Phase']] += 1
            # create new phase fraction variable
            all_variables.append(
                v.PhaseFraction(vertex['Phase'],
                                composition_sets[vertex['Phase']])
                )
            start = len(x_0)
            # default position is centroid of the simplex
            if phase_fractions is None:
                x_0.append(1.0/len(list(simplex.iterrows())))
            else:
                # use the provided guess for the phase fraction
                x_0.append(phase_fractions[m_idx])

            # add site fraction variables
            all_variables.extend(self._variables[vertex['Phase']])
            # add starting point for variable
            for varname in self._variables[vertex['Phase']]:
                print('checking '+str(varname))
                x_0.append(vertex[str(varname)])
            print(x_0)
            index_ranges.append([start, len(x_0)])

        # Define variable bounds
        bounds = [(0.0, 1.0) for x in range(len(x_0))]

        # Create master objective function
        def obj(input_x):
            "Objective function. Takes x vector as input. Returns scalar."
            objective = 0.0
            for idx, vertex in enumerate(simplex.iterrows()):
                vertex = vertex[1]
                cur_x = input_x[index_ranges[idx][0]+1:index_ranges[idx][1]]
                print('Phase: '+vertex['Phase']+' '+str(cur_x))
                # phase fraction times value of objective for that phase
                objective += input_x[index_ranges[idx][0]] * \
                    self._phase_callables[vertex['Phase']](
                        *list(cur_x))
            return objective

        # Create master gradient function
        def gradient(input_x):
            "Accepts input vector and returns gradient vector."
            gradient = np.zeros(len(input_x))
            for idx, vertex in enumerate(simplex.iterrows()):
                vertex = vertex[1]
                cur_x = input_x[index_ranges[idx][0]+1:index_ranges[idx][1]]
                # phase fraction derivative is just the phase energy
                gradient[index_ranges[idx][0]] = \
                    self._phase_callables[vertex['Phase']](
                        *list(cur_x))
                # gradient for particular phase's variables
                # NOTE: We assume all phase d.o.f are independent here,
                # and we handle any coupling through the constraints
                for g_idx, grad in \
                    enumerate(self._gradient_callables[vertex['Phase']]):
                    gradient[index_ranges[idx][0]+1+g_idx] = \
                        input_x[index_ranges[idx][0]] * \
                            grad(*list(cur_x))
            print('grad: '+str(gradient))
            return gradient

        # Generate constraint sequence
        constraints = []

        # phase fraction constraint
        def phasefrac_cons(input_x):
            "Accepts input vector and returns phase fraction constraint."
            output = sum([input_x[i[0]] for i in index_ranges]) - 1.0
            return output
        def phasefrac_jac(input_x):
            "Accepts input vector and returns Jacobian of constraint."
            output_x = np.zeros(len(input_x))
            for idx_range in index_ranges:
                output_x[idx_range[0]] = 1.0
            return output_x
        phasefrac_dict = dict()
        phasefrac_dict['type'] = 'eq'
        phasefrac_dict['fun'] = phasefrac_cons
        phasefrac_dict['jac'] = phasefrac_jac
        constraints.append(phasefrac_dict)

        # bounds constraint
        def nonneg_cons(input_x, idx, flipped):
            "Accepts input vector and returns non-negativity constraint."
            if not flipped:
                print('nonneg_cons: '+str(input_x[idx]*10000.0))
                return input_x[idx]
            else:
                print('one_cons: '+str(input_x[idx]*10000.0))
                return 1.0 - input_x[idx]
        def nonneg_jac(input_x, idx, flipped):
            "Accepts input vector and returns Jacobian of constraint."
            output_x = np.zeros(len(input_x))
            if not flipped:
                output_x[idx] = 1.0
            else:
                output_x[idx] = -1.0
            return output_x
        for idx in range(len(all_variables)):
            nonneg_dict = dict()
            nonneg_dict['type'] = 'ineq'
            nonneg_dict['fun'] = nonneg_cons
            nonneg_dict['jac'] = nonneg_jac
            nonneg_dict['args'] = [idx, False]
            #constraints.append(nonneg_dict)
            one_dict = dict()
            one_dict['type'] = 'ineq'
            one_dict['fun'] = nonneg_cons
            one_dict['jac'] = nonneg_jac
            one_dict['args'] = [idx, True]
            #constraints.append(one_dict)

        # Generate all site fraction constraints
        for idx_range in index_ranges:
            # need to generate constraint for each sublattice
            dofs = self._sublattice_dof[all_variables[idx_range[0]].phase_name]
            cur_idx = idx_range[0]+1
            for dof in dofs:
                sitefrac_dict = dict()
                sitefrac_dict['type'] = 'ineq'
                sitefrac_dict['fun'] = sitefrac_cons
                sitefrac_dict['jac'] = sitefrac_jac
                sitefrac_dict['args'] = [[cur_idx, cur_idx+dof-1]]
                cur_idx += dof-1
                if dof > 1:
                    constraints.append(sitefrac_dict)

        # All other constraints, e.g., mass balance
        def molefrac_cons(input_x, species, fix_val, all_variables, phases):
            """
            Accept input vector, species and fixed value.
            Returns constraint.
            """
            output = -fix_val
            for idx, vertex in enumerate(simplex.iterrows()):
                vertex = vertex[1]
                cur_x = input_x[index_ranges[idx][0]+1:index_ranges[idx][1]]

                res = self._molefrac_callables[vertex['Phase']][species](*cur_x)
                output += input_x[index_ranges[idx][0]] * res
            print('molefrac_cons: '+str(output))
            return output
        def molefrac_jac(input_x, species, fix_val, all_variables, phases):
            "Accepts input vector and returns Jacobian vector."
            output_x = np.zeros(len(input_x))
            for idx, vertex in enumerate(simplex.iterrows()):
                vertex = vertex[1]
                cur_x = input_x[index_ranges[idx][0]+1:index_ranges[idx][1]]
                output_x[index_ranges[idx][0]] = \
                    self._molefrac_callables[vertex['Phase']][species](
                        *list(cur_x))
                for g_idx, grad in \
                    enumerate(self._molefrac_jac_callables[vertex['Phase']][species]):
                    output_x[index_ranges[idx][0]+1+g_idx] = \
                        input_x[index_ranges[idx][0]] * \
                            grad(*list(cur_x))
            return output_x
        for condition, value in self.conditions.items():
            if isinstance(condition, v.Composition):
                # mass balance constraint for mole fraction
                molefrac_dict = dict()
                molefrac_dict['type'] = 'eq'
                molefrac_dict['fun'] = molefrac_cons
                molefrac_dict['jac'] = molefrac_jac
                molefrac_dict['args'] = \
                    [condition.species, value, all_variables, self._phases]
                constraints.append(molefrac_dict)

        # Run optimization
        res = scipy.optimize.minimize(obj, x_0, method='slsqp', jac=gradient,\
            bounds=bounds, constraints=constraints, tol=1e-10, \
            options={'iprint':6, 'ftol': 1e-12, 'eps':1e-12})
        return res

    def _build_objective_functions(self, dbf):
        "Construct objective function callables for each phase."
        for phase_name, phase_obj in self._phases.items():
            # Build the symbolic representation of the energy
            mod = Model(dbf, self.components, phase_name)
            # Construct an ordered list of the variables
            self._variables[phase_name], self._sublattice_dof[phase_name] = \
                generate_dof(phase_obj, self.components)
            molefrac_dict = dict([(x, molefrac_ast(phase_obj, x)) \
                for x in self.components if x != 'VA'])
            molefrac_jac_dict = dict()
            # Replace dependent variables with 1 - sum(other components)
            new_ast = mod.ast.subs(self.statevars)
            for idx, dof in enumerate(self._sublattice_dof[phase_name]):
                dependent_comp = v.SiteFraction(phase_name, idx, \
                    dbf.phases[phase_name].constituents[idx][dof-1])
                dep_idx = self._variables[phase_name].index(dependent_comp)
                dep_value = 1.0 - \
                    Add(*self._variables[phase_name][dep_idx-dof+1:dep_idx])
                print('dep_comp: '+str(dependent_comp))
                print('dep_value: '+str(dep_value))
                # Delete variable from consideration
                del self._variables[phase_name][dep_idx]
                # Make variable substitution
                new_ast = new_ast.subs({dependent_comp: dep_value})
                for comp in self.components:
                    if comp == 'VA':
                        continue
                    molefrac_dict[comp] = \
                        molefrac_dict[comp].subs({dependent_comp: dep_value})

            # Generate callables for the mole fractions
            for comp in self.components:
                if comp == 'VA':
                    continue
                molefrac_jac_dict[comp] = [ \
                    make_callable(molefrac_dict[comp].diff(vx), \
                    self._variables[phase_name], \
                    ) for vx in self._variables[phase_name]]
                molefrac_dict[comp] = make_callable(molefrac_dict[comp], \
                    self._variables[phase_name])

            print('self variables '+phase_name+ ': '+str(self._variables[phase_name]))
            # Build the "fast" representation of energy model
            self._phase_callables[phase_name] = \
                make_callable(new_ast.subs(self.statevars), \
                self._variables[phase_name])
            self._gradient_callables[phase_name] = [ \
                make_callable(new_ast.subs(self.statevars).diff(vx), \
                    self._variables[phase_name], \
                    ) for vx in self._variables[phase_name]]
            self._molefrac_callables[phase_name] = molefrac_dict
            self._molefrac_jac_callables[phase_name] = molefrac_jac_dict
