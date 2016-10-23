"""
eqsolver handles local energy minimization.
"""
from collections import defaultdict, OrderedDict
import copy
import itertools
import numpy as np
cimport numpy as np
import scipy.spatial
from pycalphad.constraints import mole_fraction
from pycalphad.core.sympydiff_utils import build_functions as compiled_build_functions
from pycalphad.core.constants import MIN_SITE_FRACTION, COMP_DIFFERENCE_TOL
import pycalphad.variables as v

# Maximum residual driving force (J/mol-atom) allowed for convergence
MAX_SOLVE_DRIVING_FORCE = 1e-4
# Maximum number of multi-phase solver iterations
MAX_SOLVE_ITERATIONS = 100
# Minimum energy (J/mol-atom) difference between iterations before stopping solver
MIN_SOLVE_ENERGY_PROGRESS = 1e-6


def remove_degenerate_phases(object phases, double[:,:] mole_fractions,
                             double[:,:] site_fractions, double[:] phase_fractions):
    """
    For each phase pair with composition difference below tolerance,
    eliminate phase with largest index.
    Also remove phases with phase fractions close to zero.

    Parameters
    ----------


    """
    cdef double[:,:] comp_matrix
    cdef double[:,:] comp_distances
    cdef double phfsum = 0
    cdef object redundant_phases, kept_phase, removed_phases, saved_indices
    cdef int num_phases = len(phases)
    cdef int phase_idx, sidx
    cdef int[:] indices
    for phase_idx in range(num_phases):
        if <unicode>phases[phase_idx] == <unicode>"":
            num_phases = phase_idx
            break
    # Group phases into multiple composition sets
    cdef object phase_indices = defaultdict(lambda: list())
    for phase_idx in range(num_phases):
        name = <unicode>phases[phase_idx]
        phase_indices[name].append(phase_idx)
    # Compute pairwise distances between compositions of like phases
    for name, idxs in phase_indices.items():
        indices = np.array(idxs, dtype=np.int32)
        if indices.shape[0] == 1:
            # Phase is unique
            continue
        comp_matrix = np.empty((np.max(indices)+1, mole_fractions.shape[1]))
        # The reason we don't do this based on Y fractions is because
        # of sublattice symmetry. It's very easy to detect a "miscibility gap" which is actually
        # symmetry equivalent, i.e., D([A, B] - [B, A]) > tol, but they are the same configuration.
        for idx in range(indices.shape[0]):
            comp_matrix[indices[idx], :] = mole_fractions[indices[idx], :]
        comp_distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(comp_matrix, metric='chebyshev'))
        redundant_phases = set()
        redundant_phases |= {indices[0]}
        for i in range(len(indices)):
            for j in range(i, len(indices)):
                if i == j:
                    continue
                if comp_distances[i, j] < COMP_DIFFERENCE_TOL:
                    redundant_phases |= {indices[i], indices[j]}
        redundant_phases = sorted(redundant_phases)
        kept_phase = redundant_phases[0]
        removed_phases = redundant_phases[1:]
        # Their NP values will be added to redundant_phases[0]
        # and they will be nulled out
        for redundant in removed_phases:
            phase_fractions[kept_phase] += phase_fractions[redundant]
            phases[redundant] = <unicode>''
    # Eliminate any 'fake points' that made it through the convex hull routine
    # These can show up from phases which aren't defined over all of composition space
    for phase_idx in range(num_phases):
        if <unicode>phases[phase_idx] == <unicode>'_FAKE_':
            phase_fractions[phase_idx] = np.nan
            phases[phase_idx] = <unicode>''
        else:
            # Delete unstable phases
            for phf_idx in range(phase_fractions.shape[1]):
                if phase_fractions[phf_idx] <= MIN_SITE_FRACTION:
                    phase_fractions[phase_idx] = np.nan
                    phases[phase_idx] = <unicode>''
    # Rewrite properties to delete all the nulled out phase entries
    # Then put them at the end
    # That will let us rewrite 'phases' to have only the independent phases
    # And still preserve convenient indexing with phase_idx
    saved_indices = []
    for phase_idx in range(num_phases):
        if phases[phase_idx] != '':
            saved_indices.append(phase_idx)
    saved_indices = np.array(saved_indices, dtype=np.int32)
    for sidx in saved_indices:
        # TODO: Assumes N=1 always
        phfsum += phase_fractions[sidx]
    phase_idx = 0
    for sidx in saved_indices:
        # TODO: Assumes N=1 always
        phase_fractions[phase_idx] = phase_fractions[sidx]
        phase_fractions[phase_idx] /= phfsum
        phases[phase_idx] = phases[sidx]
        mole_fractions[phase_idx, :] = mole_fractions[sidx, :]
        site_fractions[phase_idx, :] = site_fractions[sidx, :]
        phase_idx += 1
    phases[saved_indices.shape[0]:] = <unicode>''
    phase_fractions[saved_indices.shape[0]:] = np.nan
    mole_fractions[saved_indices.shape[0]:, :] = np.nan
    site_fractions[saved_indices.shape[0]:, :] = np.nan


def _compute_phase_dof(dbf, comps, phases):
    """
    Generate a list of the number of each phase's internal phase degrees of freedom.
    """
    phase_dof = []
    for name in phases:
        total = 0
        for idx in range(len(dbf.phases[name].sublattices)):
            active_in_subl = set(dbf.phases[name].constituents[idx]).intersection(comps)
            total += len(active_in_subl)
        phase_dof.append(total)
    return phase_dof

def _compute_constraints(object dbf, object comps, object phases,
                         object cur_conds, np.ndarray[dtype=np.float64_t, ndim=1] site_fracs,
                         double[:] phase_fracs, object phase_records,
                         l_multipliers=None, chempots=None, mole_fractions=None):
    """
    Compute the constraint vector and constraint Jacobian matrix.
    """
    cdef int num_sitefrac_bals = sum([len(dbf.phases[i].sublattices) for i in phases])
    cdef int num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
    cdef double indep_sum = sum([float(val) for i, val in cur_conds.items() if i.startswith('X_')])
    cdef object dependent_comp = set(comps) - set([i[2:] for i in cur_conds.keys() if i.startswith('X_')]) - {'VA'}
    dependent_comp = list(dependent_comp)[0]
    mole_fractions = mole_fractions if mole_fractions is not None else {}
    cdef int num_constraints = num_sitefrac_bals + num_mass_bals
    cdef int num_vars = len(site_fracs) + len(phases)
    cdef object phase_dof = _compute_phase_dof(dbf, comps, phases)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] l_constraints = np.zeros(num_constraints)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] chemical_potentials
    cdef np.ndarray[dtype=np.float64_t, ndim=2] constraint_jac = np.zeros((num_constraints, num_vars))
    cdef np.ndarray[dtype=np.float64_t, ndim=3] constraint_hess = np.zeros((num_constraints, num_vars, num_vars))
    cdef np.ndarray[dtype=np.uint8_t, ndim=1] contains_vacancies = np.zeros(len(phases), dtype=np.uint8)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] comp_grad_value
    cdef int phase_idx, var_offset, constraint_offset, var_idx, iter_idx
    cdef double phase_frac
    cdef bint con_vacs
    if l_multipliers is None:
        l_multipliers = np.zeros(num_constraints)
        if chempots is not None:
            l_multipliers[sum([len(dbf.phases[i].sublattices) for i in phases]):
                          sum([len(dbf.phases[i].sublattices) for i in phases]) + num_mass_bals] = chempots
    # Convenience object for caller so it doesn't need to know about the constraint configuration
    chemical_potentials = l_multipliers[sum([len(dbf.phases[i].sublattices) for i in phases]):
        sum([len(dbf.phases[i].sublattices) for i in phases]) + num_mass_bals]

    # Ordering of constraints by row: sitefrac bal of each phase, then component mass balance
    # Ordering of constraints by column: site fractions of each phase, then phase fractions
    # First: Site fraction balance constraints
    var_idx = 0
    constraint_offset = 0
    for phase_idx, name in enumerate(phases):
        for idx in range(len(dbf.phases[name].sublattices)):
            active_in_subl = set(dbf.phases[name].constituents[idx]).intersection(comps)
            if 'VA' in active_in_subl and len(active_in_subl) > 1:
                contains_vacancies[phase_idx] = True
            constraint_jac[constraint_offset + idx,
            var_idx:var_idx + len(active_in_subl)] = 1
            # print('L_CONSTRAINTS[{}] = {}'.format(constraint_offset+idx, (sum(site_fracs[var_idx:var_idx + len(active_in_subl)]) - 1)))
            l_constraints[constraint_offset + idx] = \
                (sum(site_fracs[var_idx:var_idx + len(active_in_subl)]) - 1)
            var_idx += len(active_in_subl)
        constraint_offset += len(dbf.phases[name].sublattices)
    # Second: Mass balance of each component
    for comp in [c for c in comps if c != 'VA']:
        var_offset = 0
        phase_idx = 0
        for iter_idx in range(len(phases)):
            name = phases[iter_idx]
            phase_frac = phase_fracs[iter_idx]
            con_vacs = contains_vacancies[iter_idx]
            if mole_fractions.get((name, comp), None) is None:
                mole_fractions[(name, comp)] = compiled_build_functions(mole_fraction(dbf.phases[name], comps, comp),
                                                                        sorted(set(phase_records[name].variables) - {v.T, v.P},
                                                                        key=str))
            comp_obj = mole_fractions[(name,comp)][0]
            comp_grad = mole_fractions[(name,comp)][1]
            comp_hess = mole_fractions[(name,comp)][2]
            # We use this twice, so we assign it to a variable
            comp_grad_value = comp_grad(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
            #print('MOLE FRACTIONS', (name, comp))
            # current phase frac times the comp_grad
            constraint_jac[constraint_offset,
            var_offset:var_offset + phase_dof[phase_idx]] = \
                phase_frac * comp_grad_value
            #print('CONSTRAINT_JAC[{}] += {}'.format((constraint_offset, slice(var_offset,var_offset + phase_dof[phase_idx])), phase_frac * np.squeeze(comp_grad(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))))
            constraint_jac[constraint_offset, len(site_fracs) + phase_idx] += \
                comp_obj(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
            #print('CONSTRAINT_JAC[{}] += {}'.format((constraint_offset, len(site_fracs) + phase_idx), np.squeeze(comp_obj(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))))
            # This term should only be non-zero for vacancy-containing sublattices
            # This check is to silence a warning about comp_hess() being zero
            if con_vacs:
                constraint_hess[constraint_offset,
                                var_offset:var_offset + phase_dof[phase_idx],
                                var_offset:var_offset + phase_dof[phase_idx]] = \
                    phase_frac * comp_hess(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
            constraint_hess[constraint_offset,
            var_offset:var_offset + phase_dof[phase_idx], len(site_fracs) + phase_idx] = \
            constraint_hess[constraint_offset,
            len(site_fracs) + phase_idx, var_offset:var_offset + phase_dof[phase_idx]] = comp_grad_value
            l_constraints[constraint_offset] += \
                phase_frac * comp_obj(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
            #print('L_CONSTRAINTS[{}] += {}'.format(constraint_offset, phase_frac * np.squeeze(comp_obj(*site_fracs[var_offset:var_offset+phase_dof[phase_idx]]))))
            var_offset += phase_dof[phase_idx]
            phase_idx += 1
        if comp != dependent_comp:
            l_constraints[constraint_offset] -= float(cur_conds['X_' + comp])
            #print('L_CONSTRAINTS[{}] -= {}'.format(constraint_offset, float(cur_conds['X_'+comp])))
        else:
            # TODO: Assuming N=1 (fixed for dependent component)
            l_constraints[constraint_offset] -= (1 - indep_sum)
            #print('L_CONSTRAINTS[{}] -= {}'.format(constraint_offset, (1-indep_sum)))
        #l_constraints[constraint_offset] *= -1
        # print('L_CONSTRAINTS[{}] *= -1'.format(constraint_offset))
        constraint_offset += 1
    #print('L_CONSTRAINTS', l_constraints)
    return np.array(l_constraints), np.array(constraint_jac), np.array(constraint_hess), \
        np.array(l_multipliers), np.array(chemical_potentials), mole_fractions


def _build_multiphase_gradient(dbf, comps, phases, cur_conds, site_fracs, phase_fracs,
                               l_constraints, constraint_jac, l_multipliers, callable_dict, phase_records):
    cdef double obj_result = 0
    var_offset = 0
    phase_idx = 0
    phase_dof = _compute_phase_dof(dbf, comps, phases)
    num_vars = len(site_fracs) + len(phases)
    gradient_term = np.zeros(num_vars, dtype=np.float)
    for name, phase_frac in zip(phases, phase_fracs):
        obj = callable_dict[name]
        grad = phase_records[name].grad
        obj_res = obj(*itertools.chain([cur_conds['P'], cur_conds['T']],
                                       site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
                      )
        obj_result += phase_frac * obj_res
        grad_res = grad(*itertools.chain([cur_conds['P'], cur_conds['T']],
                                         site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
                        )
        gradient_term[var_offset:var_offset + phase_dof[phase_idx]] = \
            phase_frac * np.squeeze(grad_res)[2:]  # Remove P,T grad part
        gradient_term[len(site_fracs) + phase_idx] = obj_res
        var_offset += phase_dof[phase_idx]
        phase_idx += 1
    return obj_result, gradient_term

def _build_multiphase_system(dbf, comps, phases, cur_conds, site_fracs, phase_fracs,
                             l_constraints, constraint_jac, constraint_hess, l_multipliers,
                             callable_dict, phase_records):
    # Now build objective Hessian and gradient terms
    var_offset = 0
    phase_idx = 0
    phase_dof = _compute_phase_dof(dbf, comps, phases)
    num_vars = len(site_fracs) + len(phases)
    l_hessian = np.zeros((num_vars, num_vars), dtype=np.float)
    gradient_term = np.zeros(num_vars, dtype=np.float)
    for name, phase_frac in zip(phases, phase_fracs):
        obj = callable_dict[name]
        hess = phase_records[name].hess
        grad = phase_records[name].grad
        dof = tuple(itertools.chain([np.array(cur_conds['P'], dtype=np.float), np.array(cur_conds['T'], dtype=np.float)],
                                    site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))
        obj_res = obj(*dof)
        grad_res = grad(*dof)
        gradient_term[var_offset:var_offset + phase_dof[phase_idx]] = \
            phase_frac * np.squeeze(grad_res)[2:]  # Remove P,T grad part
        gradient_term[len(site_fracs) + phase_idx] = obj_res
        hess_slice = np.index_exp[var_offset:var_offset + phase_dof[phase_idx], var_offset:var_offset + phase_dof[phase_idx]]
        l_hessian[hess_slice] = \
            phase_frac * np.squeeze(hess(*dof)
                                    )[2:, 2:]  # Remove P,T hessian part
        # Phase fraction / site fraction cross derivative
        l_hessian[len(site_fracs) + phase_idx, var_offset:var_offset + phase_dof[phase_idx]] = \
            l_hessian[var_offset:var_offset + phase_dof[phase_idx], len(site_fracs) + phase_idx] = \
            np.squeeze(grad_res)[2:] # Remove P,T grad part
        var_offset += phase_dof[phase_idx]
        phase_idx += 1
    # Constraint contribution to the Hessian (some constraints like mass balance are nonlinear)
    l_hessian -= np.multiply(l_multipliers[:, np.newaxis, np.newaxis], constraint_hess).sum(axis=0)
    return l_hessian, gradient_term

def _solve_eq_at_conditions(dbf, comps, properties, phase_records, callable_dict, conds_keys, verbose):
    """
    Compute equilibrium for the given conditions.
    This private function is meant to be called from a worker subprocess.
    For that case, usually only a small slice of the master 'properties' is provided.
    Since that slice will be copied, we also return the modified 'properties'.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    properties : Dataset
        Will be modified! Thermodynamic properties and conditions.
    phase_records : dict of PhaseRecord
        Details on phase callables.
    callable_dict : dict of callable
        Objective functions for each phase.
    conds_keys : list of str
        List of conditions axes in dimension order.
    verbose : bool
        Print details.

    Returns
    -------
    properties : Dataset
        Modified with equilibrium values.
    """
    # Factored out via profiling
    prop_MU_values = properties['MU'].values
    prop_NP_values = properties['NP'].values
    prop_Phase_values = properties['Phase'].values
    prop_X_values = properties['X'].values
    prop_Y_values = properties['Y'].values
    prop_GM_values = properties['GM'].values

    it = np.nditer(prop_GM_values, flags=['multi_index'])

    #if verbose:
    #    print('INITIAL CONFIGURATION')
    #    print(properties.MU)
    #    print(properties.Phase)
    #    print(properties.NP)
    #    print(properties.X)
    #    print(properties.Y)
    #    print('---------------------')
    while not it.finished:
        # A lot of this code relies on cur_conds being ordered!
        cur_conds = OrderedDict(zip(conds_keys,
                                    [properties['GM'].coords[b][a] for a, b in zip(it.multi_index, conds_keys)]))
        if len(cur_conds) == 0:
            cur_conds = properties['GM'].coords
        # sum of independently specified components
        indep_sum = np.sum([float(val) for i, val in cur_conds.items() if i.startswith('X_')])
        if indep_sum > 1:
            # Sum of independent component mole fractions greater than one
            # Skip this condition set
            # We silently allow this to make 2-D composition mapping easier
            prop_MU_values[it.multi_index] = np.nan
            prop_NP_values[it.multi_index + np.index_exp[:len(phases)]] = np.nan
            prop_Phase_values[it.multi_index + np.index_exp[:len(phases)]] = ''
            prop_X_values[it.multi_index + np.index_exp[:len(phases)]] = np.nan
            prop_Y_values[it.multi_index] = np.nan
            prop_GM_values[it.multi_index] = np.nan
            it.iternext()
            continue
        dependent_comp = set(comps) - set([i[2:] for i in cur_conds.keys() if i.startswith('X_')]) - {'VA'}
        if len(dependent_comp) == 1:
            dependent_comp = list(dependent_comp)[0]
        else:
            raise ValueError('Number of dependent components different from one')
        # chem_pots = OrderedDict(zip(properties.coords['component'].values, properties['MU'].values[it.multi_index]))
        # Used to cache generated mole fraction functions
        mole_fractions = {}
        for cur_iter in range(MAX_SOLVE_ITERATIONS):
            # print('CUR_ITER:', cur_iter)
            phases = list(prop_Phase_values[it.multi_index])
            if '' in phases:
                old_phase_length = phases.index('')
            else:
                old_phase_length = -1
            remove_degenerate_phases(prop_Phase_values[it.multi_index], prop_X_values[it.multi_index],
                                     prop_Y_values[it.multi_index], prop_NP_values[it.multi_index])
            phases = list(prop_Phase_values[it.multi_index])
            if '' in phases:
                new_phase_length = phases.index('')
            else:
                new_phase_length = -1
            # Are there removed phases?
            if '' in phases:
                num_phases = phases.index('')
            else:
                num_phases = len(phases)
            zero_dof = np.all(
                (prop_Y_values[it.multi_index] == 1.) | np.isnan(prop_Y_values[it.multi_index]))
            if (num_phases == 1) and zero_dof:
                # Single phase with zero internal degrees of freedom, can't do any refinement
                # TODO: In the future we may be able to refine other degrees of freedom like temperature
                # Chemical potentials have no meaning for this case
                prop_MU_values[it.multi_index] = np.nan
                break
            phases = prop_Phase_values[it.multi_index + np.index_exp[:num_phases]]
            # num_sitefrac_bals = sum([len(dbf.phases[i].sublattices) for i in phases])
            # num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
            phase_fracs = prop_NP_values[it.multi_index + np.index_exp[:len(phases)]]
            phase_dof = [len(set(phase_records[name].variables) - {v.T, v.P}) for name in phases]
            # Flatten site fractions array and remove nan padding
            site_fracs = prop_Y_values[it.multi_index].ravel()
            # That *should* give us the internal dof
            # This may break if non-padding nan's slipped in from elsewhere...
            site_fracs = site_fracs[~np.isnan(site_fracs)]
            site_fracs[site_fracs < MIN_SITE_FRACTION] = MIN_SITE_FRACTION
            phase_fracs[phase_fracs < MIN_SITE_FRACTION] = MIN_SITE_FRACTION
            var_idx = 0
            for name in phases:
                for idx in range(len(dbf.phases[name].sublattices)):
                    active_in_subl = set(dbf.phases[name].constituents[idx]).intersection(comps)
                    site_fracs[var_idx:var_idx + len(active_in_subl)] /= \
                        np.sum(site_fracs[var_idx:var_idx + len(active_in_subl)], keepdims=True)
                    var_idx += len(active_in_subl)
            # Reset Lagrange multipliers if active set of phases change
            if cur_iter == 0 or (old_phase_length != new_phase_length):
                l_multipliers = None

            l_constraints, constraint_jac, constraint_hess, l_multipliers, old_chem_pots, mole_fraction_funcs = \
                _compute_constraints(dbf, comps, phases, cur_conds, site_fracs, phase_fracs, phase_records,
                                     l_multipliers=l_multipliers,
                                     chempots=prop_MU_values[it.multi_index], mole_fractions=mole_fractions)
            qmat, rmat = np.linalg.qr(constraint_jac.T, mode='complete')
            m = rmat.shape[1]
            n = qmat.shape[0]
            # Construct orthonormal basis for the constraints
            ymat = qmat[:, :m]
            zmat = qmat[:, m:]
            # Equation 18.14a in Nocedal and Wright
            p_y = np.linalg.solve(np.dot(constraint_jac, ymat), -l_constraints)
            num_vars = len(site_fracs) + len(phases)
            l_hessian, gradient_term = _build_multiphase_system(dbf, comps, phases, cur_conds, site_fracs, phase_fracs,
                                                                l_constraints, constraint_jac, constraint_hess,
                                                                l_multipliers, callable_dict, phase_records)
            # Equation 18.18 in Nocedal and Wright
            if m != n:
                try:
                     p_z = np.linalg.solve(np.dot(np.dot(zmat.T, l_hessian), zmat),
                                           -np.dot(np.dot(np.dot(zmat.T, l_hessian), ymat), p_y) - np.dot(zmat.T, gradient_term))
                except np.linalg.LinAlgError:
                    p_z = np.zeros(zmat.shape[-1], dtype=np.float)
            else:
                zmat = np.array(0)
                p_z = 0
            step = np.dot(ymat, p_y) + np.dot(zmat, p_z)
            old_energy = copy.deepcopy(prop_GM_values[it.multi_index])
            old_chem_pots = copy.deepcopy(prop_MU_values[it.multi_index])
            candidate_site_fracs = site_fracs + step[:len(site_fracs)]
            candidate_site_fracs[candidate_site_fracs < MIN_SITE_FRACTION] = MIN_SITE_FRACTION
            candidate_site_fracs[candidate_site_fracs > 1] = 1
            candidate_phase_fracs = phase_fracs + \
                                    step[len(candidate_site_fracs):len(candidate_site_fracs) + len(phases)]
            candidate_phase_fracs[candidate_phase_fracs < MIN_SITE_FRACTION] = 0
            candidate_phase_fracs[candidate_phase_fracs > 1] = 1
            (candidate_l_constraints, candidate_constraint_jac, candidate_constraint_hess,
             candidate_l_multipliers, candidate_chem_pots, mole_fraction_funcs) = \
                _compute_constraints(dbf, comps, phases, cur_conds,
                                     candidate_site_fracs, candidate_phase_fracs, phase_records,
                                     l_multipliers=l_multipliers, mole_fractions=mole_fractions)
            candidate_energy, candidate_gradient_term = \
                _build_multiphase_gradient(dbf, comps, phases, cur_conds, candidate_site_fracs,
                                           candidate_phase_fracs, candidate_l_constraints,
                                           candidate_constraint_jac, candidate_l_multipliers,
                                           callable_dict, phase_records)
            # We updated degrees of freedom this iteration
            new_l_multipliers = np.linalg.solve(np.dot(constraint_jac, ymat).T,
                                                np.dot(ymat.T, gradient_term + np.dot(l_hessian, step)))
            # XXX: Should fix underlying numerical problem at edges of composition space instead of working around
            if np.any(np.isnan(new_l_multipliers)) or np.any(np.abs(new_l_multipliers) > 1e10):
                if verbose:
                    print('WARNING: Unstable Lagrange multipliers: ', new_l_multipliers)
                # Equation 18.16 in Nocedal and Wright
                # This method is less accurate but more stable
                new_l_multipliers = np.dot(np.dot(np.linalg.inv(np.dot(candidate_constraint_jac,
                                                                       candidate_constraint_jac.T)),
                                           candidate_constraint_jac), candidate_gradient_term)
            l_multipliers = new_l_multipliers
            if verbose:
                print('NEW_L_MULTIPLIERS', l_multipliers)
            num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
            chemical_potentials = l_multipliers[sum([len(dbf.phases[i].sublattices) for i in phases]):
                                                sum([len(dbf.phases[i].sublattices) for i in phases]) + num_mass_bals]
            prop_MU_values[it.multi_index] = chemical_potentials
            prop_NP_values[it.multi_index + np.index_exp[:len(phases)]] = candidate_phase_fracs
            prop_X_values[it.multi_index + np.index_exp[:len(phases)]] = 0
            prop_GM_values[it.multi_index] = candidate_energy
            var_offset = 0
            for phase_idx in range(len(phases)):
                prop_Y_values[it.multi_index + np.index_exp[phase_idx, :phase_dof[phase_idx]]] = \
                    candidate_site_fracs[var_offset:var_offset + phase_dof[phase_idx]]
                for comp_idx, comp in enumerate([c for c in comps if c != 'VA']):
                    prop_X_values[it.multi_index + np.index_exp[phase_idx, comp_idx]] = \
                        mole_fraction_funcs[(phases[phase_idx], comp)][0](
                            *candidate_site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
                var_offset += phase_dof[phase_idx]

            properties.attrs['solve_iterations'] += 1
            total_comp = np.nansum(prop_NP_values[it.multi_index][..., np.newaxis] * \
                                   prop_X_values[it.multi_index], axis=-2)
            driving_force = (prop_MU_values[it.multi_index] * total_comp).sum(axis=-1) - \
                             prop_GM_values[it.multi_index]
            driving_force = np.squeeze(driving_force)
            if verbose:
                print('Chem pot progress', prop_MU_values[it.multi_index] - old_chem_pots)
                print('Energy progress', prop_GM_values[it.multi_index] - old_energy)
                print('Driving force', driving_force)
            no_progress = np.abs(prop_MU_values[it.multi_index] - old_chem_pots).max() < 0.01
            no_progress &= np.abs(prop_GM_values[it.multi_index] - old_energy) < MIN_SOLVE_ENERGY_PROGRESS
            if no_progress and np.abs(driving_force) > MAX_SOLVE_DRIVING_FORCE:
                print('Driving force failed to converge: {}'.format(cur_conds))
                prop_MU_values[it.multi_index] = np.nan
                prop_NP_values[it.multi_index] = np.nan
                prop_X_values[it.multi_index] = np.nan
                prop_Y_values[it.multi_index] = np.nan
                prop_GM_values[it.multi_index] = np.nan
                prop_Phase_values[it.multi_index] = ''
                break
            elif no_progress:
                if verbose:
                    print('No progress')
                num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
                chemical_potentials = l_multipliers[sum([len(dbf.phases[i].sublattices) for i in phases]):
                                                    sum([len(dbf.phases[i].sublattices) for i in phases]) + num_mass_bals]
                prop_MU_values[it.multi_index] = chemical_potentials
                break
            elif (not no_progress) and cur_iter == MAX_SOLVE_ITERATIONS-1:
                print('Failed to converge: {}'.format(cur_conds))
                prop_MU_values[it.multi_index] = np.nan
                prop_NP_values[it.multi_index] = np.nan
                prop_X_values[it.multi_index] = np.nan
                prop_Y_values[it.multi_index] = np.nan
                prop_GM_values[it.multi_index] = np.nan
                prop_Phase_values[it.multi_index] = ''
        it.iternext()
    return properties