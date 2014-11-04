"""
The minimize module handles calculation of equilibrium.
"""
from __future__ import division
from pycalphad import Model
import pycalphad.variables as v
from pycalphad.theanocode import theano_function
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import itertools
try:
    set
except NameError:
    from sets import Set as set #pylint: disable=W0622

def point_sample(comp_count, size=10):
    """
    Sample 'size' ** len('comp_count') points in composition space
    for the sublattice configuration specified by 'comp_count'.
    Points are sampled psuedo-randomly from a symmetric Dirichlet distribution.
    Note: For sublattices with only one component, only one point will be
        returned, regardless of 'size'.

    Parameters
    ----------
    comp_count : list
        Number of components in each sublattice.
    size : int
        Number of points to sample _per sublattice_.

    Returns
    -------
    ndarray of generated points satisfying the mass balance.

    Examples
    --------
    >>> comps = [8,1] # 8 components in sublattice 1; only 1 in sublattice 2
    >>> pts = point_sample(comps, size=20)
    """
    subl_points = []
    for ctx in comp_count:
        if ctx > 1:
            pts = np.random.dirichlet(tuple(np.ones(ctx)), size)
            subl_points.append(pts)
        elif ctx == 1:
            # only 1 component; no degrees of freedom
            subl_points.append(np.atleast_2d(1))
        else:
            raise ValueError('Number of components must be >= 1')

    # Calculate Cartesian product over all sublattices
    #pylint: disable=W0142
    prod = itertools.product(*subl_points)
    # join together the coordinates in each sublattice
    result = list(map(np.concatenate, prod))
    return np.asarray(result)

class CompositionSet(object):
    """
    CompositionSets are used by equilibrium calculations to represent
    phases during the calculation.

    Attributes
    ----------
    ast, SymPy object
        Abstract representation of energy function
    energy, Theano function
        Compiled energy function optimized for call speed
    variables, list
        Input variables, ordered in the way 'energy' expects

    Methods
    -------
    None yet.

    Examples
    --------
    None yet.
    """
    def __init__(self, mod, statevars, variables):
        self.ast = mod.ast.subs(statevars)
        self.variables = variables
        print(self.variables)
        self.energy = theano_function(self.variables, [self.ast])

def eq(db, comps, phases, **kwargs):
    """
    Calculate the equilibrium state of a system containing the specified
    components and phases. Model parameters are taken from 'db' and any
    state variables (T, P, etc.) can be specified as keyword arguments.

    Parameters
    ----------
    db : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names (case-sensitive) of components to consider in the calculation.
    phases : list
        Names (case-sensitive) of phases to consider in the calculation.

    Returns
    -------
    Structured equilibrium calculation.

    Examples
    --------
    None yet.
    """
    # Here we would check for any keyword arguments that are special, i.e.,
    # there may be keyword arguments that aren't state variables

    points_per_phase = 10000
    # Convert keyword strings to proper state variable objects
    # If we don't do this, sympy will get confused during substitution
    statevars = {v.StateVariable(k): val for k, val in kwargs.items()}
    # Consider only the active phases
    active_phases = {name: db.phases[name] for name in phases}
    comp_sets = {}
    # Construct a dataframe to hold all the data
    all_phases_df = pd.DataFrame()
    for phase_name, phase_obj in active_phases.items():
        # Build the symbolic representation of the energy
        mod = Model(db, comps, phase_name)
        # Construct an ordered list of the variables
        variables = []
        for idx, sublattice in enumerate(phase_obj.constituents):
            for component in sorted(sublattice):
                variables.append(v.SiteFraction(phase_name, idx, component))

        # Build the "fast" representation of that model
        comp_sets[phase_name] = CompositionSet(mod, statevars, variables)

        # Make user-friendly site fraction column labels
        var_names = ['Y('+variable.phase_name+',' + \
                str(variable.sublattice_index) + ',' + variable.species +')' \
                for variable in variables]

        # Calculate the number of components in each sublattice
        sublattice_dof = list(map(len, phase_obj.constituents))
        nontrivial_sublattices = len(sublattice_dof) - sublattice_dof.count(1)
        # Get the site ratios in each sublattice
        site_ratios = list(phase_obj.sublattices)
        # Choose a sensible number of compositions to sample
        num_points = int(points_per_phase**(1/nontrivial_sublattices))
        # Sample composition space
        points = point_sample(sublattice_dof, size=num_points)
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
        for idx, p in enumerate(points):
            energies[idx] = comp_sets[phase_name].energy(*p)

        # Add points and calculated energies to the DataFrame
        data_dict = {'GM':energies, 'Phase':phase_name}
        data_dict.update(statevars)

        for comp in sorted(comps):
            if comp == 'VA':
                continue
            data_dict['X('+comp+')'] = [0 for n in range(len(points))]

        for column_idx, data in enumerate(points.T):
            data_dict[var_names[column_idx]] = data

        # Now map the internal degrees of freedom to global coordinates
        for p_idx, p in enumerate(points):
            for idx, coordinate in enumerate(p):
                cur_var = variables[idx]
                if cur_var.species == 'VA':
                    continue
                ratio = site_ratios[cur_var.sublattice_index]
                data_dict['X('+cur_var.species+')'][p_idx] += ratio*coordinate

        phase_df = pd.DataFrame(data_dict)
        # Merge dataframe into master dataframe
        # TODO: Better way to do this than concat (always copies)?
        all_phases_df = \
            pd.concat([all_phases_df, phase_df], axis=0, join='outer', \
                        ignore_index=True)
    # all_phases_df now contains energy surface information for the system
    return all_phases_df

def find_hull(points, energies, sublattice_dof):
    # Construct a matrix to calculate the convex hull of this phase
    gibbs_matrix = np.ndarray((len(points), sum(sublattice_dof)+1))
    gibbs_matrix[:, 0:-1] = points
    gibbs_matrix[:, -1] = energies

    # Strip out the dependent degrees of freedom before finding the hull
    all_dof = set(range(sum(sublattice_dof)))
    cur_idx = 0
    dependent_dof = set()
    for dof in sublattice_dof:
        cur_idx += dof
        dependent_dof.append(cur_idx-1)
    independent_dof = list(all_dof - dependent_dof)
    internal_hull = ConvexHull(gibbs_matrix[:, independent_dof])

