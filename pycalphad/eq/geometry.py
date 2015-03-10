"""
The geometry module handles geometric calculations associated with
equilibrium calculation.
"""

import pycalphad.variables as v
from pycalphad.eq.simplex import Simplex
import numpy as np
import itertools

def lower_convex_hull(data, comps, conditions):
    """
    Find the simplex on the lower convex hull satisfying the specified
    conditions.

    Parameters
    ----------
    data : DataFrame
        A sample of the energy surface of the system.
    comps : list
        All the components in the system.
    conditions : dict or (list of dict)
        StateVariables and their corresponding value.

    Returns
    -------
    A list of indices corresponding to vertices of the simplex.
    If 'conditions' is a list, this will return a list (in 'conditions' order)
    of a list of indices.
    Note: This routine will not check if the simplex is degenerate.

    Examples
    --------
    None yet.
    """
    # determine column indices for degrees of freedom
    dof = ['X({0})'.format(c) for c in comps if c != 'VA']
    dof_values = []
    for cond, value in conditions.items():
        if not isinstance(cond, v.Composition):
            continue
        # ignore phase-specific composition conditions
        if cond.phase_name is not None:
            continue
        if cond.species == 'VA':
            continue
        dof_values.append(value)

    dof.append('GM')
    dof_values.append(1-sum(dof_values))
    dof_values = np.array(dof_values)

    # convert DataFrame of independent columns to ndarray
    dat = data[dof].values

    # Build a fictitious hyperplane which has an energy greater than the max
    # energy in the system
    # This guarantees our starting point is feasible but
    # also guarantees it won't be part of the solution
    energy_ceiling = np.amax(dat[:, -1]) + 1
    start_matrix = np.empty([len(dof)-1, len(dof)])
    start_matrix[:, :-1] = np.eye(len(dof)-1)
    start_matrix[:, -1] = energy_ceiling # set energy
    dat = np.concatenate([start_matrix, dat])

    max_iterations = 1000
    # Need to choose a feasible starting point
    # initialize simplex as first n points of fictitious hyperplane
    candidate_simplex = list(range(len(dof)-1))
    # Calculate chemical potentials
    candidate_potentials = np.linalg.solve(dat[candidate_simplex, :-1],
                                           dat[candidate_simplex, -1])

    # Calculate driving forces for reducing our candidate potentials
    driving_forces = dat[:, -1] - np.dot(dat[:, :-1], candidate_potentials)
    new_points = np.where(driving_forces < 1e-4)[0]
    # Don't test points in the candidate simplex
    new_points = np.delete(new_points, candidate_simplex)
    print(new_points)
    candidate_energy = np.dot(candidate_potentials, dof_values)
    iteration = 0
    found_solution = False

    while (found_solution == False) and (iteration < max_iterations):
        iteration += 1
        for new_point in new_points.copy():
            found_point = False
            # Need to successively replace columns with the new point
            # The goal is to find positive phase fraction values
            fractions = np.empty(len(dof_values))
            new_simplex = np.empty(dat.shape[1] - 1, dtype=np.int)
            for col in range(dat.shape[1] - 1):
                print(candidate_simplex)
                new_simplex[:] = candidate_simplex # deep copy
                new_simplex[col] = new_point
                print(new_simplex)
                test_matrix = dat[new_simplex, :-1]
                fractions[:-1] = np.linalg.solve(test_matrix[:-1, :-1] - \
                    test_matrix[-1, :-1], dof_values[:-1] - \
                    test_matrix[-1, :-1])
                fractions[-1] = 1 - sum(fractions[:-1])
                print(fractions)
                if np.all(fractions > -1e-8):
                    # Positive phase fractions
                    # Do I reduce the energy with this solution?
                    # Recalculate chemical potentials and energy
                    new_potentials = np.linalg.solve(dat[new_simplex, :-1],
                                                     dat[new_simplex, -1])
                    new_energy = np.dot(new_potentials, dof_values)
                    if new_energy < candidate_energy:
                        print('lol')
                        candidate_simplex[:] = new_simplex
                        candidate_potentials[:] = new_potentials
                        candidate_energy = new_energy
                        # Recalculate driving forces
                        driving_forces[:] = dat[:, -1] - \
                            np.dot(dat[:, :-1], candidate_potentials)
                        new_points = np.where(driving_forces < 1e-4)[0]
                        # Don't test points in the candidate simplex
                        new_points = np.delete(new_points, candidate_simplex)
                        found_point = True
                        break
            if found_point:
                break
        # If there is no positive driving force, we have the solution
        if np.any(new_points) == False:
            found_solution = True
            print('Solution:')
            print(dat[candidate_simplex])
            print(candidate_potentials)
            print(candidate_energy)
            return candidate_simplex

    print('Iterations exceeded')
    print(new_points)
    return None
