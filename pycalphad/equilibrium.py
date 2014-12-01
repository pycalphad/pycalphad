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
from sympy import diff
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
        tri = scipy.spatial.Delaunay(hull.points[:, :-1])
        candidate_simplex = tri.find_simplex([independent_dof_values])
        return self.data.iloc[tri.simplices[candidate_simplex][0]]


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

        # Create master objective function
        def obj(input_x):
            "Objective function. Takes x vector as input. Returns scalar."
            objective = 0
            for idx, vertex in enumerate(simplex.T.to_dict().values()):
                cur_x = input_x[index_ranges[idx][0]+1:index_ranges[idx][1]]
                # phase fraction times value of objective for that phase
                objective += index_ranges[idx][0] * \
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
                    gradient[index_ranges[idx][0]+g_idx] = \
                        input_x[index_ranges[idx][0]] * \
                            grad(*(list(self.statevars.values()) +list(cur_x)))
            return gradient
        # Generate constraint dictionary

        # Run optimization
        print(scipy.optimize.minimize(obj, x_0, jac=gradient, bounds=bounds))


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
            self._gradient_callables[phase_name] = \
                make_callable(list(mod.gradient.values()), \
                list(self.statevars.keys()) + \
                    self._variables[phase_name], mode=ast)
            # Make user-friendly site fraction column labels
            self._varnames[phase_name] = ['Y('+variable.phase_name+',' + \
                    str(variable.sublattice_index) + ',' + \
                    variable.species +')' \
                        for variable in self._variables[phase_name]]

