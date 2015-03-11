"""
The geometry module handles geometric calculations associated with
equilibrium calculation.
"""

import pycalphad.variables as v
import numpy as np

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
    comps = sorted(list(comps))
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
    # This guarantees our starting point is feasible but also guarantees
    # it won't be part of the solution
    energy_ceiling = np.amax(dat[:, -1]) + 1
    start_matrix = np.empty([len(dof)-1, len(dof)])
    start_matrix[:, :-1] = np.eye(len(dof)-1)
    start_matrix[:, -1] = energy_ceiling # set energy
    dat = np.concatenate([start_matrix, dat])

    max_iterations = 100
    # Need to choose a feasible starting point
    # initialize simplex as first n points of fictitious hyperplane
    candidate_simplex = np.array(range(len(dof)-1), dtype=np.int)
    # Calculate chemical potentials
    candidate_potentials = np.linalg.solve(dat[candidate_simplex, :-1],
                                           dat[candidate_simplex, -1])

    # Calculate driving forces for reducing our candidate potentials
    driving_forces = np.dot(dat[:, :-1], candidate_potentials) - dat[:, -1]
    # Mask points with negative (or nearly zero) driving force
    point_mask = driving_forces < 1e-4
    #print(point_mask)
    #print(np.array(range(dat.shape[0]), dtype=np.int)[~point_mask])
    candidate_energy = np.dot(candidate_potentials, dof_values)
    fractions = np.empty(len(dof_values))
    iteration = 0
    found_solution = False

    while (found_solution == False) and (iteration < max_iterations):
        iteration += 1
        for new_point in np.array(range(dat.shape[0]), dtype=np.int)[~point_mask]:
            found_point = False
            # Need to successively replace columns with the new point
            # The goal is to find positive phase fraction values
            new_simplex = np.empty(dat.shape[1] - 1, dtype=np.int)
            for col in range(dat.shape[1] - 1):
                #print(candidate_simplex)
                new_simplex[:] = candidate_simplex # [:] forces copy
                new_simplex[col] = new_point
                #print(new_simplex)
                fractions = np.linalg.solve(dat[new_simplex, :-1].T, dof_values)
                #print(fractions)
                if np.all(fractions > -1e-8):
                    # Positive phase fractions
                    # Do I reduce the energy with this solution?
                    # Recalculate chemical potentials and energy
                    #print('new matrix: {0}'.format(dat[new_simplex, :-1]))
                    #print('new energies: {0}'.format(dat[new_simplex, -1]))
                    new_potentials = np.linalg.solve(dat[new_simplex, :-1],
                                                     dat[new_simplex, -1])
                    #print('new_potentials: {0}'.format(new_potentials))
                    new_energy = np.dot(new_potentials, dof_values)
                    if new_energy < candidate_energy:
                        #print('New simplex {2} reduces energy from {0} to {1}' \
                        #    .format(candidate_energy, new_energy, new_simplex))
                        # [:] notation forces a copy
                        candidate_simplex[:] = new_simplex
                        candidate_potentials[:] = new_potentials
                        candidate_energy = new_energy
                        # Recalculate driving forces with new potentials
                        driving_forces[:] = np.dot(dat[:, :-1], \
                            candidate_potentials) - dat[:, -1]
                        point_mask = driving_forces < 1e-4
                        # Don't test points on the fictitious hyperplane
                        point_mask[list(range(len(dof)-1))] = True
                        found_point = True
                        break
                    #else:
                    #    print('New simplex {2} increases energy from {0} to {1}' \
                    #        .format(candidate_energy, new_energy, new_simplex))
            if found_point:
                #print('Found feasible simplex: moving to next iteration')
                break
            #else:
            #    print('{0} is not feasible'.format(new_point))
            #    print('Driving force: {0}'.format(driving_forces[new_point]))
        # If there is no positive driving force, we have the solution
        #print('Checking point mask')
        #print(point_mask)
        if np.all(point_mask) == True:
            found_solution = True
            print('Solution:')
            print(dat[candidate_simplex])
            #print(candidate_potentials)
            print(candidate_energy)
            #print(fractions)
            # Fix candidate simplex indices to remove fictitious points
            candidate_simplex = candidate_simplex - (len(dof)-1)
            check = candidate_simplex < 0
            if not np.any(check):
                return candidate_simplex, fractions

    print('Iterations exceeded')
    print('Positive driving force still exists for these points')
    print(np.where(driving_forces > 1e-4)[0])
    return None, None
