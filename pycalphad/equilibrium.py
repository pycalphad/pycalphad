"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""

import pycalphad.variables as v
from pycalphad.minimize import point_sample, make_callable
from pycalphad import Model
import pandas as pd
import numpy as np
import scipy.spatial
import scipy.optimize
from collections import Counter
import copy

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
    points_per_phase : int, optional
        Approximate number of points to sample per phase.
    ast : ['theano', 'numpy'], optional
        Specify how we should construct the callable for the energy.

    Returns
    -------
    Structured equilibrium calculation.

    Examples
    --------
    None yet.
    """
    def __init__(self, dbf, comps, phases, conditions,
                 points_per_phase=10000, ast='numpy', **kwargs):
        self.conditions = conditions
        self.components = set(comps)
        self.phases = dict()
        self.statevars = dict()
        self.data = pd.DataFrame()

        self._phases = dict([[name, dbf.phases[name]] for name in phases])
        self._phase_callables = dict()
        self._gradient_callables = dict()
        self._variables = dict()
        self._varnames = dict()
        self._sublattice_dof = dict()
        # Here we would check for any keyword arguments that are special, i.e.,
        # there may be keyword arguments that aren't state variables

        # Convert keyword strings to proper state variable objects
        # If we don't do this, sympy will get confused during substitution
        # TODO: Needs to differentiate T,P from N when specifying conditions
        self.statevars = \
            dict((v.StateVariable(key), value) \
                for (key, value) in kwargs.items())

        self._build_objective_functions(dbf, ast)

        for phase_obj in [dbf.phases[name] for name in phases]:
            data_dict = \
                self._calculate_energy_surface(phase_obj, points_per_phase)

            phase_df = pd.DataFrame(data_dict)
            # TODO: Apply phase-specific conditions
            # Merge dataframe into master dataframe
            self.data = \
                pd.concat([self.data, phase_df], axis=0, join='outer', \
                            ignore_index=True)
        # self.data now contains energy surface information for the system
        # find simplex for a starting point; refine with optimization
        refined_values = self.minimize(self.get_starting_simplex())
        print(refined_values)

    def get_starting_simplex(self):
        "Calculate convex hull and find a suitable starting point."
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

        independent_dof.append('GM')
        # calculate the convex hull for the independent d.o.f
        # TODO: Apply activity conditions here
        hull = scipy.spatial.ConvexHull(
            self.data[independent_dof].values
        )
        # locate the simplex closest to our desired condition
        tri = scipy.spatial.Delaunay(hull.points[hull.vertices][:, :-1])
        candidate_simplex = tri.find_simplex([independent_dof_values])
        return self.data.iloc[hull.vertices[tri.simplices[candidate_simplex][0]]]


    def minimize(self, simplex):
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
        for vertex in simplex.T.to_dict().values():
            # increase multiplicity by one
            composition_sets[vertex['Phase']] += 1
            # create new phase fraction variable
            all_variables.append(
                v.PhaseFraction(vertex['Phase'],
                                composition_sets[vertex['Phase']])
                )
            start = len(x_0)
            # default position is centroid of the simplex
            x_0.append(1/len(list(simplex.iterrows())))

            # add site fraction variables
            all_variables.extend(self._variables[vertex['Phase']])
            # add starting point for variable
            for varname in self._varnames[vertex['Phase']]:
                x_0.append(vertex[varname])

            index_ranges.append([start, len(x_0)])

        # Define variable bounds
        bounds = [[0, 1]] * len(x_0)
        print(all_variables)
        print(x_0)

        # Create master objective function
        def obj(input_x):
            "Objective function. Takes x vector as input. Returns scalar."
            objective = 0
            for idx, vertex in enumerate(simplex.T.to_dict().values()):
                cur_x = input_x[index_ranges[idx][0]+1:index_ranges[idx][1]]
                # phase fraction times value of objective for that phase
                objective += input_x[index_ranges[idx][0]] * \
                    self._phase_callables[vertex['Phase']](
                        *(list(self.statevars.values()) + list(cur_x)))
            return objective

        # Create master gradient function
        def gradient(input_x):
            "Accepts input vector and returns gradient vector."
            gradient = np.zeros(len(input_x))
            for idx, vertex in enumerate(simplex.T.to_dict().values()):
                cur_x = input_x[index_ranges[idx][0]+1:index_ranges[idx][1]]
                # phase fraction derivative is just the phase energy
                gradient[index_ranges[idx][0]] = \
                    self._phase_callables[vertex['Phase']](
                        *(list(self.statevars.values()) + list(cur_x)))
                # gradient for particular phase's variables
                # NOTE: We assume all phase d.o.f are independent here,
                # and we handle any coupling through the constraints
                for g_idx, grad in \
                    enumerate(self._gradient_callables[vertex['Phase']]):
                    gradient[index_ranges[idx][0]+1+g_idx] = \
                        input_x[index_ranges[idx][0]] * \
                            grad(*(list(self.statevars.values()) +list(cur_x)))
            return gradient

        # Generate constraint sequence
        constraints = []

        # phase fraction constraint
        def phasefrac_cons(input_x):
            "Accepts input vector and returns phase fraction constraint."
            output = sum([input_x[i[0]] for i in index_ranges]) - 1
            return output
        def phasefrac_jac(input_x):
            "Accepts input vector and returns Jacobian of constraint."
            output_x = np.zeros(len(input_x))
            for idx_range in index_ranges:
                output_x[idx_range[0]] = 1
            return output_x
        phasefrac_dict = dict()
        phasefrac_dict['type'] = 'eq'
        phasefrac_dict['fun'] = phasefrac_cons
        phasefrac_dict['jac'] = phasefrac_jac
        constraints.append(phasefrac_dict)

        # site fraction constraints
        # we use idx_range to force a range of input_x to sum to 1
        def sitefrac_cons(input_x, idx_range):
            """
            Accepts input vector and index range and returns
            site fraction constraint.
            """
            return sum(input_x[idx_range[0]:idx_range[1]]) - 1
        def sitefrac_jac(input_x, idx_range):
            """
            Accepts input vector and index range and returns
            Jacobian of site fraction constraint.
            """
            output_x = np.zeros(len(input_x))
            for idx in range(idx_range[0], idx_range[1]):
                output_x[idx] = 1
            return output_x
        # Generate all site fraction constraints
        for idx_range in index_ranges:
            # need to generate constraint for each sublattice
            dofs = self._sublattice_dof[all_variables[idx_range[0]].phase_name]
            cur_idx = idx_range[0]+1
            for dof in dofs:
                sitefrac_dict = dict()
                sitefrac_dict['type'] = 'eq'
                sitefrac_dict['fun'] = sitefrac_cons
                sitefrac_dict['jac'] = sitefrac_jac
                sitefrac_dict['args'] = [[cur_idx, cur_idx+dof]]
                cur_idx += dof
                constraints.append(sitefrac_dict)

        def molefrac_cons(input_x, species, fix_val):
            """
            Accept input vector, species and fixed value.
            Returns constraint.
            """
            output = -fix_val
            phase_idx = 0
            site_ratios = []
            for idx, variable in enumerate(all_variables):
                if isinstance(variable, v.PhaseFraction):
                    phase_idx = idx
                    # Normalize site ratios
                    site_ratio_normalization = 0
                    for n_idx, sublattice in enumerate(self._phases[variable.phase_name].constituents):
                        if species in set(sublattice):
                            site_ratio_normalization += self._phases[variable.phase_name].sublattices[n_idx]
            
                    site_ratios = [c/site_ratio_normalization for c in self._phases[variable.phase_name].sublattices]
                if isinstance(variable, v.SiteFraction) and \
                    species == variable.species:
                    #print(str(all_variables[phase_idx]) +'*'+ \
                    #    str(site_ratios[variable.sublattice_index])+'*'+str(all_variables[idx]))
                    output += input_x[phase_idx] * \
                        site_ratios[variable.sublattice_index] * input_x[idx]
            return output

        def molefrac_jac(input_x, species, fix_val):
            """
            Accept input vector, species and fixed value.
            Returns Jacobian of constraint.
            """
            output_x = np.zeros(len(input_x))
            phase_idx = 0
            site_ratios = []
            for idx, variable in enumerate(all_variables):
                if isinstance(variable, v.PhaseFraction):
                    phase_idx = idx
                    # Normalize site ratios
                    site_ratio_normalization = 0
                    for n_idx, sublattice in enumerate(self._phases[variable.phase_name].constituents):
                        if species in set(sublattice):
                            site_ratio_normalization += self._phases[variable.phase_name].sublattices[n_idx]
            
                    site_ratios = [c/site_ratio_normalization \
                        for c in self._phases[variable.phase_name].sublattices]
                    # We add the phase fraction Jacobian contribution below
                if isinstance(variable, v.SiteFraction) and \
                    species == variable.species:
                    output_x[idx] += input_x[phase_idx] * \
                        site_ratios[variable.sublattice_index]
                    output_x[phase_idx] += input_x[idx] * \
                        site_ratios[variable.sublattice_index]
            return output_x

        # All other constraints, e.g., mass balance
        for condition, value in self.conditions.items():
            if isinstance(condition, v.Composition):
                # mass balance constraint for mole fraction
                molefrac_dict = dict()
                molefrac_dict['type'] = 'eq'
                molefrac_dict['fun'] = molefrac_cons
                molefrac_dict['jac'] = molefrac_jac
                molefrac_dict['args'] = [condition.species, value]
                constraints.append(molefrac_dict)

        # Run optimization
        res = scipy.optimize.minimize(obj, x_0, method='SLSQP', jac=gradient, \
        bounds=bounds, constraints=constraints, options={'disp':True})
        return res


    def _calculate_energy_surface(self, phase_obj, points_per_phase):
        "Sample the energy surface of a phase and return a data dictionary"
        # Calculate the number of components in each sublattice
        nontrivial_sublattices = \
            len(self._sublattice_dof[phase_obj.name]) - \
                self._sublattice_dof[phase_obj.name].count(1)
        # Get the site ratios in each sublattice
        site_ratios = list(phase_obj.sublattices)
        # Choose a sensible number of compositions to sample
        num_points = None
        if nontrivial_sublattices > 0:
            num_points = int(points_per_phase**(1/nontrivial_sublattices))
        else:
            # Fixed stoichiometry
            num_points = 1
        # Sample composition space
        points = point_sample(self._sublattice_dof[phase_obj.name],
                              size=num_points)
        # Allocate space for energies, once calculated
        energies = np.zeros(len(points))

        # Normalize site ratios
        site_ratio_normalization = 0
        for idx, sublattice in enumerate(phase_obj.constituents):
            # sublattices with only vacancies don't count
            if len(sublattice) == 1 and sublattice[0] == 'VA':
                continue
            site_ratio_normalization += site_ratios[idx]

        site_ratios = [c/site_ratio_normalization for c in site_ratios]

        # TODO: not very efficient point sampling strategy
        # in principle, this could be parallelized
        for idx, point in enumerate(points):
            energies[idx] = \
                self._phase_callables[phase_obj.name](
                    *(list(self.statevars.values()) + list(point))
                )

        # Add points and calculated energies to the DataFrame
        data_dict = {'GM':energies, 'Phase':phase_obj.name}
        data_dict.update(self.statevars)

        for comp in sorted(list(self.components)):
            if comp == 'VA':
                continue
            data_dict['X('+comp+')'] = [0 for n in range(len(points))]

        for column_idx, data in enumerate(points.T):
            data_dict[self._varnames[phase_obj.name][column_idx]] = data

        # Now map the internal degrees of freedom to global coordinates
        for p_idx, pts in enumerate(points):
            for idx, coordinate in enumerate(pts):
                cur_var = self._variables[phase_obj.name][idx]
                if cur_var.species == 'VA':
                    continue
                ratio = site_ratios[cur_var.sublattice_index]
                data_dict['X('+cur_var.species+')'][p_idx] += ratio*coordinate

        return data_dict

    def _build_objective_functions(self, dbf, ast):
        "Construct objective function callables for each phase."
        for phase_name, phase_obj in self._phases.items():
            # Build the symbolic representation of the energy
            mod = Model(dbf, list(self.components), phase_name)
            # Construct an ordered list of the variables
            self._variables[phase_name] = []
            sublattice_dof = []
            for idx, sublattice in enumerate(phase_obj.constituents):
                dof = 0
                for component in set(sublattice).intersection(self.components):
                    self._variables[phase_name].append(
                        v.SiteFraction(phase_name, idx, component)
                        )
                    dof += 1
                sublattice_dof.append(dof)
            self._sublattice_dof[phase_name] = sublattice_dof
            # Build the "fast" representation of that model
            self._phase_callables[phase_name] = make_callable(mod.ast, \
                list(self.statevars.keys()) + \
                    self._variables[phase_name], mode=ast)
            self._gradient_callables[phase_name] = [make_callable(grad_ast, \
                    list(self.statevars.keys()) + self._variables[phase_name],\
                    mode=ast) for grad_ast in list(mod.gradient.values())]
            # Make user-friendly site fraction column labels
            self._varnames[phase_name] = ['Y('+variable.phase_name+',' + \
                    str(variable.sublattice_index) + ',' + \
                    variable.species +')' \
                        for variable in self._variables[phase_name]]

