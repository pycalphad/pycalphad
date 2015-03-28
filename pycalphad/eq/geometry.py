"""
The geometry module handles geometric calculations associated with
equilibrium calculation.
"""

import pycalphad.variables as v
import numpy as np
from pycalphad.log import logger

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
    conditions : dict
        StateVariables and their corresponding value.

    Returns
    -------
    A tuple containing:
    (1) A numpy array of indices corresponding to vertices of the simplex.
    (2) A numpy array corresponding to the phase fractions.
    Note: This routine will not check if the simplex is degenerate.

    Examples
    --------
    None yet.
    """
    # determine column indices for degrees of freedom
    comps = sorted(list(comps))
    dof = ['X({0})'.format(c) for c in comps if c != 'VA']
    dof_values = np.zeros(len(dof))
    marked_dof_values = list(range(len(dof)))
    for cond, value in conditions.items():
        if not isinstance(cond, v.Composition):
            continue
        # ignore phase-specific composition conditions
        if cond.phase_name is not None:
            continue
        if cond.species == 'VA':
            continue
        dof_values[comps.index(cond.species)] = value
        marked_dof_values.remove(comps.index(cond.species))

    dof.append('GM')

    if len(marked_dof_values) == 1:
        dof_values[marked_dof_values[0]] = 1-sum(dof_values)
    else:
        logger.error('Not enough composition conditions specified')
        raise ValueError('Not enough composition conditions specified.')

    # convert DataFrame of independent columns to ndarray
    dat = data[dof].values

    # Build a fictitious hyperplane which has an energy greater than the max
    # energy in the system
    # This guarantees our starting point is feasible but also makes it likely
    # it won't be part of the solution
    energy_ceiling = np.amax(dat[:, -1])
    if energy_ceiling < 0:
        energy_ceiling *= 0.1
    else:
        energy_ceiling *= 10
    start_matrix = np.empty([len(dof)-1, len(dof)])
    start_matrix[:, :-1] = np.eye(len(dof)-1)
    start_matrix[:, -1] = energy_ceiling # set energy
    dat = np.concatenate([start_matrix, dat])

    max_iterations = dat.shape[0]
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
    #logger.debug(point_mask)
    #logger.debug(np.array(range(dat.shape[0]), dtype=np.int)[~point_mask])
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
                logger.debug('trial matrix: %s', dat[new_simplex, :-1].T)
                try:
                    fractions = np.linalg.solve(dat[new_simplex, :-1].T,
                                                dof_values)
                except np.linalg.LinAlgError:
                    # singular matrix means the trial simplex is degenerate
                    # this usually happens due to collisions between points on
                    # the fictitious hyperplane and the endmembers
                    continue
                logger.debug('fractions: %s', fractions)
                if np.all(fractions > -1e-8):
                    # Positive phase fractions
                    # Do I reduce the energy with this solution?
                    # Recalculate chemical potentials and energy
                    logger.debug('new matrix: {0}'.format(dat[new_simplex, :-1]))
                    logger.debug('new energies: {0}'.format(dat[new_simplex, -1]))
                    new_potentials = np.linalg.solve(dat[new_simplex, :-1],
                                                     dat[new_simplex, -1])
                    logger.debug('new_potentials: {0}'.format(new_potentials))
                    new_energy = np.dot(new_potentials, dof_values)
                    # differences of less than 1mJ/mol are irrelevant
                    new_energy = np.around(new_energy, decimals=3)
                    if new_energy <= candidate_energy:
                        logger.debug('New simplex {2} reduces energy from \
                            {0} to {1}'.format(candidate_energy, new_energy, \
                            new_simplex))
                        # [:] notation forces a copy
                        candidate_simplex[:] = new_simplex
                        candidate_potentials[:] = new_potentials
                        # np.array() forces a copy
                        candidate_energy = np.array(new_energy)
                        # Recalculate driving forces with new potentials
                        driving_forces[:] = np.dot(dat[:, :-1], \
                            candidate_potentials) - dat[:, -1]
                        logger.debug('driving_forces: %s', driving_forces)
                        point_mask = driving_forces < 1e-4
                        # Don't test points on the fictitious hyperplane
                        point_mask[list(range(len(dof)-1))] = True
                        found_point = True
                        break
                    else:
                        logger.debug('Trial simplex {2} increases energy from {0} to {1}'\
                                    .format(candidate_energy, new_energy, new_simplex))
                        logger.debug('%s points with positive driving force remain',
                                     list(driving_forces >= 1e-4).count(True))
            if found_point:
                logger.debug('Found feasible simplex: moving to next iteration')
                logger.debug('%s points with positive driving force remain',
                             list(driving_forces >= 1e-4).count(True))
                break
            #else:
            #    print('{0} is not feasible'.format(new_point))
            #    print('Driving force: {0}'.format(driving_forces[new_point]))
        # If there is no positive driving force, we have the solution
        #print('Checking point mask')
        #print(point_mask)
        logger.debug('Iteration count: {0}'.format(iteration))
        if np.all(point_mask) == True:
            logger.debug('Unadjusted candidate_simplex: %s', candidate_simplex)
            logger.debug(dat[candidate_simplex])
            # Fix candidate simplex indices to remove fictitious points
            candidate_simplex = candidate_simplex - (len(dof)-1)
            logger.debug('Adjusted candidate_simplex: %s', candidate_simplex)
            # Remove fictitious points from the candidate simplex
            # These can inadvertently show up if we only calculate a phase with
            # limited solubility
            # Also remove points with very small estimated phase fractions
            candidate_simplex, fractions = zip(*[(c, f) for c, f in
                                                 zip(candidate_simplex,
                                                     fractions)
                                                 if c >= 0 and f >= 1e-12])
            candidate_simplex = np.array(candidate_simplex)
            fractions = np.array(fractions)
            fractions /= np.sum(fractions)
            logger.debug('Final candidate_simplex: %s', candidate_simplex)
            logger.debug('Final phase fractions: %s', fractions)
            found_solution = True
            logger.debug('Solution:')
            logger.debug(candidate_potentials)
            logger.debug(candidate_energy)
            return candidate_simplex, fractions

    logger.error('Iterations exceeded')
    logger.debug('Positive driving force still exists for these points')
    logger.debug(np.where(driving_forces > 1e-4)[0])
    return None, None
