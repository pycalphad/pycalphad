"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""
from __future__ import print_function
import pycalphad.variables as v
from pycalphad.core.utils import unpack_kwarg
from pycalphad.core.utils import unpack_condition, unpack_phases
from pycalphad import calculate, Model
from pycalphad.constraints import mole_fraction
from pycalphad.core.lower_convex_hull import lower_convex_hull
from pycalphad.core.sympydiff_utils import build_functions as compiled_build_functions
from pycalphad.core.constants import MIN_SITE_FRACTION, COMP_DIFFERENCE_TOL
from sympy import Add, Symbol
import dask
from dask import delayed
import dask.multiprocessing, dask.async
from xarray import Dataset, DataArray
import numpy as np
import scipy.spatial
from collections import defaultdict, namedtuple, OrderedDict
import itertools
import copy
from datetime import datetime

#def delayed(func, *fargs, **fkwargs):
#    return func

# Maximum number of multi-phase solver iterations
MAX_SOLVE_ITERATIONS = 100
# Minimum energy (J/mol-atom) difference between iterations before stopping solver
MIN_SOLVE_ENERGY_PROGRESS = 1e-6
# Maximum residual driving force (J/mol-atom) allowed for convergence
MAX_SOLVE_DRIVING_FORCE = 1e-4

PhaseRecord = namedtuple('PhaseRecord', ['variables', 'grad', 'hess', 'plane_grad', 'plane_hess',
                                         'mass_obj', 'mass_grad', 'mass_hess'])

class EquilibriumError(Exception):
    "Exception related to calculation of equilibrium"
    pass


class ConditionError(EquilibriumError):
    "Exception related to equilibrium conditions"
    pass


def remove_degenerate_phases(properties, multi_index):
    """
    For each phase pair with composition difference below tolerance,
    eliminate phase with largest index.
    Also remove phases with phase fractions close to zero.

    Parameters
    ----------
    properties : xarray.Dataset
        Equilibrium calculation data. This will be modified!
    multi_index : tuple
        Index into 'properties' of the condition set of interest.

    """
    # Factored out via profiling
    prop_Phase = properties.Phase.values
    prop_X = properties.X.values
    prop_Y = properties.Y.values
    prop_NP = properties.NP.values

    phases = list(prop_Phase[multi_index])
    # Are there already removed phases?
    if '' in phases:
        num_phases = phases.index('')
    else:
        num_phases = len(phases)
    phases = prop_Phase[multi_index + np.index_exp[:num_phases]]
    # Group phases into multiple composition sets
    phase_indices = defaultdict(lambda: list())
    for phase_idx, name in enumerate(phases):
        phase_indices[name].append(phase_idx)
    # Compute pairwise distances between compositions of like phases
    for name, indices in phase_indices.items():
        if len(indices) == 1:
            # Phase is unique
            continue
        # The reason we don't do this based on Y fractions is because
        # of sublattice symmetry. It's very easy to detect a "miscibility gap" which is actually
        # symmetry equivalent, i.e., D([A, B] - [B, A]) > tol, but they are the same configuration.
        comp_matrix = prop_X[multi_index + np.index_exp[indices]]
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
            prop_NP[multi_index + np.index_exp[kept_phase]] += \
                prop_NP[multi_index + np.index_exp[redundant]]
            prop_Phase[multi_index + np.index_exp[redundant]] = ''
    # Eliminate any 'fake points' that made it through the convex hull routine
    # These can show up from phases which aren't defined over all of composition space
    prop_NP[np.nonzero(prop_Phase == '_FAKE_')] = np.nan
    prop_Phase[np.nonzero(prop_Phase == '_FAKE_')] = ''
    # Delete unstable phases
    unstable_phases = np.nonzero(prop_NP[multi_index] <= MIN_SITE_FRACTION)
    prop_Phase[multi_index + np.index_exp[unstable_phases]] = ''
    # Rewrite properties to delete all the nulled out phase entries
    # Then put them at the end
    # That will let us rewrite 'phases' to have only the independent phases
    # And still preserve convenient indexing of 'properties' with phase_idx
    saved_indices = prop_Phase[multi_index] != ''
    saved_indices = np.arange(len(saved_indices))[saved_indices]
    # TODO: Assumes N=1 always
    prop_NP[multi_index + np.index_exp[:len(saved_indices)]] = \
        prop_NP[multi_index + np.index_exp[saved_indices]] / \
        np.sum(prop_NP[multi_index + np.index_exp[saved_indices]])
    prop_NP[multi_index + np.index_exp[len(saved_indices):]] = np.nan
    prop_Phase[multi_index + np.index_exp[:len(saved_indices)]] = \
        prop_Phase[multi_index + np.index_exp[saved_indices]]
    prop_Phase[multi_index + np.index_exp[len(saved_indices):]] = ''
    prop_X[multi_index + np.index_exp[:len(saved_indices), :]] = \
        prop_X[multi_index + np.index_exp[saved_indices, :]]
    prop_X[multi_index + np.index_exp[len(saved_indices):, :]] = np.nan
    prop_Y[multi_index + np.index_exp[:len(saved_indices), :]] = \
        prop_Y[multi_index + np.index_exp[saved_indices, :]]
    prop_Y[multi_index + np.index_exp[len(saved_indices):, :]] = np.nan


def _adjust_conditions(conds):
    "Adjust conditions values to be within the numerical limit of the solver."
    new_conds = OrderedDict()
    for key, value in sorted(conds.items(), key=str):
        if isinstance(key, v.Composition):
            new_conds[key] = [max(val, MIN_SITE_FRACTION*1000) for val in unpack_condition(value)]
        else:
            new_conds[key] = unpack_condition(value)
    return new_conds

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

def _compute_constraints(dbf, comps, phases, cur_conds, site_fracs, phase_fracs, phase_records,
                         l_multipliers=None, chempots=None, mole_fractions=None):
    """
    Compute the constraint vector and constraint Jacobian matrix.
    """
    num_sitefrac_bals = sum([len(dbf.phases[i].sublattices) for i in phases])
    num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
    indep_sum = np.sum([float(val) for i, val in cur_conds.items() if i.startswith('X_')])
    dependent_comp = set(comps) - set([i[2:] for i in cur_conds.keys() if i.startswith('X_')]) - {'VA'}
    dependent_comp = list(dependent_comp)[0]
    mole_fractions = mole_fractions if mole_fractions is not None else {}
    num_constraints = num_sitefrac_bals + num_mass_bals
    num_vars = len(site_fracs) + len(phases)
    phase_dof = _compute_phase_dof(dbf, comps, phases)
    l_constraints = np.zeros(num_constraints, dtype=np.float)
    if l_multipliers is None:
        l_multipliers = np.zeros(num_constraints, dtype=np.float)
        if chempots is not None:
            l_multipliers[sum([len(dbf.phases[i].sublattices) for i in phases]):
                          sum([len(dbf.phases[i].sublattices) for i in phases]) + num_mass_bals] = chempots
    # Convenience object for caller so it doesn't need to know about the constraint configuration
    chemical_potentials = l_multipliers[sum([len(dbf.phases[i].sublattices) for i in phases]):
        sum([len(dbf.phases[i].sublattices) for i in phases]) + num_mass_bals]

    constraint_jac = np.zeros((num_constraints, num_vars), dtype=np.float)
    constraint_hess = np.zeros((num_constraints, num_vars, num_vars), dtype=np.float)
    contains_vacancies = np.zeros(len(phases), dtype=np.bool)
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
        for name, phase_frac, con_vacs in zip(phases, phase_fracs, contains_vacancies):
            if mole_fractions.get((name, comp), None) is None:
                mole_fractions[(name, comp)] = compiled_build_functions(mole_fraction(dbf.phases[name], comps, comp),
                                                                        sorted(set(phase_records[name].variables) - {v.T, v.P},
                                                                        key=str))
            comp_obj, comp_grad, comp_hess = mole_fractions[(name, comp)]
            #print('MOLE FRACTIONS', (name, comp))
            # current phase frac times the comp_grad
            constraint_jac[constraint_offset,
            var_offset:var_offset + phase_dof[phase_idx]] = \
                phase_frac * np.squeeze(comp_grad(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))
            #print('CONSTRAINT_JAC[{}] += {}'.format((constraint_offset, slice(var_offset,var_offset + phase_dof[phase_idx])), phase_frac * np.squeeze(comp_grad(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))))
            constraint_jac[constraint_offset, len(site_fracs) + phase_idx] += \
                np.squeeze(comp_obj(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))
            #print('CONSTRAINT_JAC[{}] += {}'.format((constraint_offset, len(site_fracs) + phase_idx), np.squeeze(comp_obj(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))))
            # This term should only be non-zero for vacancy-containing sublattices
            # This check is to silence a warning about comp_hess() being zero
            if con_vacs:
                constraint_hess[constraint_offset,
                                var_offset:var_offset + phase_dof[phase_idx],
                                var_offset:var_offset + phase_dof[phase_idx]] = \
                    phase_frac * np.squeeze(comp_hess(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))
            constraint_hess[constraint_offset,
            var_offset:var_offset + phase_dof[phase_idx], len(site_fracs) + phase_idx] = \
            constraint_hess[constraint_offset,
            len(site_fracs) + phase_idx, var_offset:var_offset + phase_dof[phase_idx]] = \
                np.squeeze(comp_grad(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))
            l_constraints[constraint_offset] += \
                phase_frac * np.squeeze(comp_obj(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))
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
    return l_constraints, constraint_jac, constraint_hess, l_multipliers, chemical_potentials, mole_fractions

def _compute_multiphase_objective(dbf, comps, phases, cur_conds, site_fracs, phase_fracs, callable_dict):
    result = 0
    phase_dof = _compute_phase_dof(dbf, comps, phases)
    var_offset = 0
    for phase_idx, (name, phase_frac) in enumerate(zip(phases, phase_fracs)):
        obj = callable_dict[name]
        obj_res = obj(*itertools.chain([cur_conds['P'], cur_conds['T']],
                                       site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
                      )
        result += phase_frac * obj_res
        var_offset += phase_dof[phase_idx]
    return result

def _build_multiphase_gradient(dbf, comps, phases, cur_conds, site_fracs, phase_fracs,
                               l_constraints, constraint_jac, l_multipliers, callable_dict, phase_records):
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
        grad_res = grad(*itertools.chain([cur_conds['P'], cur_conds['T']],
                                         site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
                        )
        gradient_term[var_offset:var_offset + phase_dof[phase_idx]] = \
            phase_frac * np.squeeze(grad_res)[2:]  # Remove P,T grad part
        gradient_term[len(site_fracs) + phase_idx] = obj_res
        var_offset += phase_dof[phase_idx]
        phase_idx += 1
    return gradient_term

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
    it = np.nditer(properties['GM'].values, flags=['multi_index'])
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
            properties['MU'].values[it.multi_index] = np.nan
            properties['NP'].values[it.multi_index + np.index_exp[:len(phases)]] = np.nan
            properties['Phase'].values[it.multi_index + np.index_exp[:len(phases)]] = ''
            properties['X'].values[it.multi_index + np.index_exp[:len(phases)]] = np.nan
            properties['Y'].values[it.multi_index] = np.nan
            properties['GM'].values[it.multi_index] = np.nan
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
            phases = list(properties['Phase'].values[it.multi_index])
            if '' in phases:
                old_phase_length = phases.index('')
            else:
                old_phase_length = -1
            remove_degenerate_phases(properties, it.multi_index)
            phases = list(properties['Phase'].values[it.multi_index])
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
                (properties['Y'].values[it.multi_index] == 1.) | np.isnan(properties['Y'].values[it.multi_index]))
            if (num_phases == 1) and zero_dof:
                # Single phase with zero internal degrees of freedom, can't do any refinement
                # TODO: In the future we may be able to refine other degrees of freedom like temperature
                # Chemical potentials have no meaning for this case
                properties['MU'].values[it.multi_index] = np.nan
                break
            phases = properties['Phase'].values[it.multi_index + np.index_exp[:num_phases]]
            # num_sitefrac_bals = sum([len(dbf.phases[i].sublattices) for i in phases])
            # num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
            phase_fracs = properties['NP'].values[it.multi_index + np.index_exp[:len(phases)]]
            phase_dof = [len(set(phase_records[name].variables) - {v.T, v.P}) for name in phases]
            # Flatten site fractions array and remove nan padding
            site_fracs = properties['Y'].values[it.multi_index].ravel()
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
                                     chempots=properties['MU'].values[it.multi_index], mole_fractions=mole_fractions)
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
            old_energy = copy.deepcopy(properties['GM'].values[it.multi_index])
            old_chem_pots = copy.deepcopy(properties['MU'].values[it.multi_index])
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
            candidate_gradient_term = _build_multiphase_gradient(dbf, comps, phases,
                                                                 cur_conds, candidate_site_fracs,
                                                                 candidate_phase_fracs,
                                                                 candidate_l_constraints, candidate_constraint_jac,
                                                                 candidate_l_multipliers, callable_dict, phase_records)
            candidate_energy = _compute_multiphase_objective(dbf, comps, phases, cur_conds, candidate_site_fracs,
                                                             candidate_phase_fracs,
                                                             callable_dict)
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
            properties['MU'].values[it.multi_index] = chemical_potentials
            properties['NP'].values[it.multi_index + np.index_exp[:len(phases)]] = candidate_phase_fracs
            properties['X'].values[it.multi_index + np.index_exp[:len(phases)]] = 0
            properties['GM'].values[it.multi_index] = candidate_energy
            var_offset = 0
            for phase_idx in range(len(phases)):
                properties['Y'].values[it.multi_index + np.index_exp[phase_idx, :phase_dof[phase_idx]]] = \
                    candidate_site_fracs[var_offset:var_offset + phase_dof[phase_idx]]
                for comp_idx, comp in enumerate([c for c in comps if c != 'VA']):
                    properties['X'].values[it.multi_index + np.index_exp[phase_idx, comp_idx]] = \
                        mole_fraction_funcs[(phases[phase_idx], comp)][0](
                            *candidate_site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
                var_offset += phase_dof[phase_idx]

            properties.attrs['solve_iterations'] += 1
            total_comp = np.nansum(properties['NP'].values[it.multi_index][..., np.newaxis] * \
                                   properties['X'].values[it.multi_index], axis=-2)
            driving_force = (properties['MU'].values[it.multi_index] * total_comp).sum(axis=-1) - \
                             properties['GM'].values[it.multi_index]
            driving_force = np.squeeze(driving_force)
            if verbose:
                print('Chem pot progress', properties['MU'].values[it.multi_index] - old_chem_pots)
                print('Energy progress', properties['GM'].values[it.multi_index] - old_energy)
                print('Driving force', driving_force)
            no_progress = np.abs(properties['MU'].values[it.multi_index] - old_chem_pots).max() < 0.01
            no_progress &= np.abs(properties['GM'].values[it.multi_index] - old_energy) < MIN_SOLVE_ENERGY_PROGRESS
            if no_progress and np.abs(driving_force) > MAX_SOLVE_DRIVING_FORCE:
                print('Driving force failed to converge: {}'.format(cur_conds))
                properties['MU'].values[it.multi_index] = np.nan
                properties['NP'].values[it.multi_index] = np.nan
                properties['X'].values[it.multi_index] = np.nan
                properties['Y'].values[it.multi_index] = np.nan
                properties['GM'].values[it.multi_index] = np.nan
                properties['Phase'].values[it.multi_index] = ''
                break
            elif no_progress:
                if verbose:
                    print('No progress')
                num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
                chemical_potentials = l_multipliers[sum([len(dbf.phases[i].sublattices) for i in phases]):
                                                    sum([len(dbf.phases[i].sublattices) for i in phases]) + num_mass_bals]
                properties['MU'].values[it.multi_index] = chemical_potentials
                break
            elif (not no_progress) and cur_iter == MAX_SOLVE_ITERATIONS-1:
                print('Failed to converge: {}'.format(cur_conds))
                properties['MU'].values[it.multi_index] = np.nan
                properties['NP'].values[it.multi_index] = np.nan
                properties['X'].values[it.multi_index] = np.nan
                properties['Y'].values[it.multi_index] = np.nan
                properties['GM'].values[it.multi_index] = np.nan
                properties['Phase'].values[it.multi_index] = ''
        it.iternext()
    return properties


def _postprocess_properties(grid, properties, conds, indep_vals):
    indexer = []
    for idx, vals in enumerate(indep_vals):
        indexer.append(np.arange(len(vals), dtype=np.int)[idx * (np.newaxis,) + np.index_exp[:] + \
                                                          (len(conds.keys()) - idx + 1) * (np.newaxis,)])
    indexer.append(properties['points'].values[..., np.newaxis])
    indexer.append(
        np.arange(grid['X'].values.shape[-1], dtype=np.int)[(len(conds.keys())) * (np.newaxis,) + np.index_exp[:]])
    ravelled_X_view = grid['X'].values[tuple(indexer)]
    indexer[-1] = np.arange(grid['Y'].values.shape[-1], dtype=np.int)[
        (len(conds.keys())) * (np.newaxis,) + np.index_exp[:]]
    ravelled_Y_view = grid['Y'].values[tuple(indexer)]
    indexer = []
    for idx, vals in enumerate(indep_vals):
        indexer.append(np.arange(len(vals), dtype=np.int)[idx * (np.newaxis,) + np.index_exp[:] + \
                                                          (len(conds.keys()) - idx) * (np.newaxis,)])
    indexer.append(properties['points'].values)
    ravelled_Phase_view = grid['Phase'].values[tuple(indexer)]
    properties['X'].values[...] = ravelled_X_view
    properties['Y'].values[...] = ravelled_Y_view
    # TODO: What about invariant reactions? We should perform a final driving force calculation here.
    # We can handle that in the same post-processing step where we identify single-phase regions.
    properties['Phase'].values[...] = ravelled_Phase_view
    del properties['points']
    return properties

def _merge_property_slices(properties, chunk_grid, slices, conds_keys, results):
    "Merge back together slices of 'properties'"
    for prop_slice, prop_arr in zip(chunk_grid, results):
        if not isinstance(prop_arr, Dataset):
            print('Error: {}'.format(prop_arr))
            continue
        all_coords = dict(zip(conds_keys, [np.atleast_1d(sl)[ch]
                                                               for ch, sl in zip(prop_slice, slices)]))
        for dv in properties.data_vars.keys():
            # Have to be very careful with how we assign to 'properties' here
            # We may accidentally assign to a copy unless we index the data variable first
            dv_coords = {key: val for key, val in all_coords.items() if key in properties[dv].coords.keys()}
            properties[dv][dv_coords] = prop_arr[dv]
    return properties

def _eqcalculate(dbf, comps, phases, conditions, output, data=None, per_phase=False, **kwargs):
    """
    WARNING: API/calling convention not finalized.
    Compute the *equilibrium value* of a property.
    This function differs from `calculate` in that it computes
    thermodynamic equilibrium instead of randomly sampling the
    internal degrees of freedom of a phase.
    Because of that, it's slower than `calculate`.
    This plugs in the equilibrium phase and site fractions
    to compute a thermodynamic property defined in a Model.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list or dict
        Names of phases to consider in the calculation.
    conditions : dict or (list of dict)
        StateVariables and their corresponding value.
    output : str
        Equilibrium model property (e.g., CPM, HM, etc.) to compute.
        This must be defined as an attribute in the Model class of each phase.
    data : Dataset, optional
        Previous result of call to `equilibrium`.
        Should contain the equilibrium configurations at the conditions of interest.
        If the databases are not the same as in the original calculation,
        the results may be meaningless. If None, `equilibrium` will be called.
        Specifying this keyword argument can save the user some time if several properties
        need to be calculated in succession.
    per_phase : bool, optional
        If True, compute and return the property for each phase present.
        If False, return the total system value, weighted by the phase fractions.
    kwargs
        Passed to `calculate`.

    Returns
    -------
    Dataset of property as a function of equilibrium conditions
    """
    if data is None:
        data = equilibrium(dbf, comps, phases, conditions)
    active_phases = unpack_phases(phases) or sorted(dbf.phases.keys())
    conds = _adjust_conditions(conditions)
    indep_vars = ['P', 'T']
    # TODO: Rewrite this to use the coord dict from 'data'
    str_conds = OrderedDict((str(key), value) for key, value in conds.items())
    indep_vals = list([float(x) for x in np.atleast_1d(val)]
                      for key, val in str_conds.items() if key in indep_vars)
    coord_dict = str_conds.copy()
    components = [x for x in sorted(comps) if not x.startswith('VA')]
    coord_dict['vertex'] = np.arange(len(components))
    grid_shape = np.meshgrid(*coord_dict.values(),
                             indexing='ij', sparse=False)[0].shape
    prop_shape = grid_shape
    prop_dims = list(str_conds.keys()) + ['vertex']

    result = Dataset({output: (prop_dims, np.full(prop_shape, np.nan))}, coords=coord_dict)
    # For each phase select all conditions where that phase exists
    # Perform the appropriate calculation and then write the result back
    for phase in active_phases:
        dof = sum([len(x) for x in dbf.phases[phase].constituents])
        current_phase_indices = (data.Phase.values == phase)
        if ~np.any(current_phase_indices):
            continue
        points = data.Y.values[np.nonzero(current_phase_indices)][..., :dof]
        statevar_indices = np.nonzero(current_phase_indices)[:len(indep_vals)]
        statevars = {key: np.take(np.asarray(vals), idx)
                     for key, vals, idx in zip(indep_vars, indep_vals, statevar_indices)}
        statevars.update(kwargs)
        if statevars.get('mode', None) is None:
            statevars['mode'] = 'numpy'
        calcres = calculate(dbf, comps, [phase], output=output,
                            points=points, broadcast=False, **statevars)
        result[output].values[np.nonzero(current_phase_indices)] = calcres[output].values
    if not per_phase:
        result[output] = (result[output] * data['NP']).sum(dim='vertex', skipna=True)
    else:
        result['Phase'] = data['Phase'].copy()
        result['NP'] = data['NP'].copy()
    return result

def equilibrium(dbf, comps, phases, conditions, output=None, model=None,
                verbose=False, broadcast=True, calc_opts=None,
                scheduler=dask.async.get_sync, **kwargs):
    """
    Calculate the equilibrium state of a system containing the specified
    components and phases, under the specified conditions.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list or dict
        Names of phases to consider in the calculation.
    conditions : dict or (list of dict)
        StateVariables and their corresponding value.
    output : str or list of str, optional
        Additional equilibrium model properties (e.g., CPM, HM, etc.) to compute.
        These must be defined as attributes in the Model class of each phase.
    model : Model, a dict of phase names to Model, or a seq of both, optional
        Model class to use for each phase.
    verbose : bool, optional
        Print details of calculations. Useful for debugging.
    broadcast : bool
        If True, broadcast conditions against each other. This will compute all combinations.
        If False, each condition should be an equal-length list (or single-valued).
        Disabling broadcasting is useful for calculating equilibrium at selected conditions,
        when those conditions don't comprise a grid.
    calc_opts : dict, optional
        Keyword arguments to pass to `calculate`, the energy/property calculation routine.
    scheduler : Dask scheduler, optional
        Job scheduler for performing the computation.
        If None, return a Dask graph of the computation instead of actually doing it.

    Returns
    -------
    Structured equilibrium calculation, or Dask graph if scheduler=None.

    Examples
    --------
    None yet.
    """
    if not broadcast:
        raise NotImplementedError('Broadcasting cannot yet be disabled')
    from pycalphad import __version__ as pycalphad_version
    active_phases = unpack_phases(phases) or sorted(dbf.phases.keys())
    comps = sorted(comps)
    if len(set(comps) - set(dbf.elements)) > 0:
        raise EquilibriumError('Components not found in database: {}'.format(','.join(set(comps) - set(dbf.elements))))
    indep_vars = ['T', 'P']
    calc_opts = calc_opts if calc_opts is not None else dict()
    model = model if model is not None else Model
    phase_records = dict()
    callable_dict = kwargs.pop('callables', dict())
    grad_callable_dict = kwargs.pop('grad_callables', dict())
    hess_callable_dict = kwargs.pop('hess_callables', dict())
    maximum_internal_dof = 0
    # Modify conditions values to be within numerical limits, e.g., X(AL)=0
    # Also wrap single-valued conditions with lists
    conds = _adjust_conditions(conditions)
    for cond in conds.keys():
        if isinstance(cond, (v.Composition, v.ChemicalPotential)) and cond.species not in comps:
            raise ConditionError('{} refers to non-existent component'.format(cond))
    str_conds = OrderedDict((str(key), value) for key, value in conds.items())
    num_calcs = np.prod([len(i) for i in str_conds.values()])
    build_functions = compiled_build_functions
    backend_mode = 'compiled'
    if kwargs.get('_backend', None):
        backend_mode = kwargs['_backend']
    if verbose:
        backend_dict = {'compiled': 'Compiled (autowrap)', 'interpreted': 'Interpreted (autograd)'}
        print('Calculation Backend: {}'.format(backend_dict.get(backend_mode, 'Custom')))
    indep_vals = list([float(x) for x in np.atleast_1d(val)]
                      for key, val in str_conds.items() if key in indep_vars)
    components = [x for x in sorted(comps) if not x.startswith('VA')]
    # Construct models for each phase; prioritize user models
    models = unpack_kwarg(model, default_arg=Model)
    if verbose:
        print('Components:', ' '.join(comps))
        print('Phases:', end=' ')
    max_phase_name_len = max(len(name) for name in active_phases)
    for name in active_phases:
        mod = models[name]
        if isinstance(mod, type):
            models[name] = mod = mod(dbf, comps, name)
        site_fracs = mod.site_fractions
        variables = sorted([v.P, v.T] + site_fracs, key=str)
        maximum_internal_dof = max(maximum_internal_dof, len(site_fracs))
        out = models[name].energy
        if (not callable_dict.get(name, False)) or not (grad_callable_dict.get(name, False)) \
                or (not hess_callable_dict.get(name, False)):
            undefs = list(out.atoms(Symbol) - out.atoms(v.StateVariable))
            for undef in undefs:
                out = out.xreplace({undef: float(0)})
            cf, gf, hf = build_functions(out, [v.P, v.T] + site_fracs)
            if callable_dict.get(name, None) is None:
                callable_dict[name] = cf
            if grad_callable_dict.get(name, None) is None:
                grad_callable_dict[name] = gf
            if hess_callable_dict.get(name, None) is None:
                hess_callable_dict[name] = hf

        # Adjust gradient by the approximate chemical potentials
        hyperplane = Add(*[v.MU(i)*mole_fraction(dbf.phases[name], comps, i)
                           for i in comps if i != 'VA'])
        mu_dof = [v.MU(i) for i in comps if i != 'VA'] + site_fracs
        plane_obj, plane_grad, plane_hess = build_functions(hyperplane,
                                                            mu_dof)
        molefracs = Add(*[mole_fraction(dbf.phases[name], comps, i)
                        for i in comps if i != 'VA'])
        mass_obj, mass_grad, mass_hess = build_functions(molefracs, site_fracs)
        phase_records[name.upper()] = PhaseRecord(variables=variables,
                                                  grad=grad_callable_dict[name],
                                                  hess=hess_callable_dict[name],
                                                  plane_grad=plane_grad,
                                                  plane_hess=plane_hess,
                                                  mass_obj=mass_obj,
                                                  mass_grad=mass_grad,
                                                  mass_hess=mass_hess)
        if verbose:
            print(name, end=' ')
    if verbose:
        print('[done]', end='\n')

    # 'calculate' accepts conditions through its keyword arguments
    grid_opts = calc_opts.copy()
    grid_opts.update({key: value for key, value in str_conds.items() if key in indep_vars})
    if 'pdens' not in grid_opts:
        grid_opts['pdens'] = 300

    coord_dict = str_conds.copy()
    coord_dict['vertex'] = np.arange(len(components))
    grid_shape = np.meshgrid(*coord_dict.values(),
                             indexing='ij', sparse=False)[0].shape
    coord_dict['component'] = components

    grid = delayed(calculate, pure=False)(dbf, comps, active_phases, output='GM',
                                          model=models, callables=callable_dict, fake_points=True, **grid_opts)

    properties = delayed(Dataset, pure=False)({'NP': (list(str_conds.keys()) + ['vertex'],
                                                      np.empty(grid_shape)),
                                               'GM': (list(str_conds.keys()),
                                                      np.empty(grid_shape[:-1])),
                                               'MU': (list(str_conds.keys()) + ['component'],
                                                      np.empty(grid_shape)),
                                               'X': (list(str_conds.keys()) + ['vertex', 'component'],
                                                     np.empty(grid_shape + (grid_shape[-1],))),
                                               'Y': (list(str_conds.keys()) + ['vertex', 'internal_dof'],
                                                     np.empty(grid_shape + (maximum_internal_dof,))),
                                               'Phase': (list(str_conds.keys()) + ['vertex'],
                                                         np.empty(grid_shape, dtype='U%s' % max_phase_name_len)),
                                               'points': (list(str_conds.keys()) + ['vertex'],
                                                          np.empty(grid_shape, dtype=np.int))
                                               },
                                              coords=coord_dict,
                                              attrs={'hull_iterations': 1, 'solve_iterations': 0,
                                                     'engine': 'pycalphad %s' % pycalphad_version},
                                              )
    # One last call to ensure 'properties' and 'grid' are consistent with one another
    properties = delayed(lower_convex_hull, pure=False)(grid, properties, verbose=verbose)
    properties = delayed(_postprocess_properties, pure=False)(grid, properties, conds, indep_vals)
    conditions_per_chunk_per_axis = 2
    if num_calcs > 1:
        # Generate slices of 'properties'
        slices = []
        for val in grid_shape[:-1]:
            idx_arr = list(range(val))
            num_chunks = int(np.floor(val/conditions_per_chunk_per_axis))
            if num_chunks > 0:
                cond_slices = [x for x in np.array_split(np.asarray(idx_arr), num_chunks) if len(x) > 0]
            else:
                cond_slices = [idx_arr]
            slices.append(cond_slices)
        chunk_dims = [len(slc) for slc in slices]
        chunk_grid = np.array(np.unravel_index(np.arange(np.prod(chunk_dims)), chunk_dims)).T
        res = []
        for chunk in chunk_grid:
            prop_slice = properties[OrderedDict(list(zip(str_conds.keys(),
                                                         [np.atleast_1d(sl)[ch] for ch, sl in zip(chunk, slices)])))]
            job = delayed(_solve_eq_at_conditions, pure=False)(dbf, comps, prop_slice,
                                                                    phase_records, callable_dict,
                                                                    list(str_conds.keys()), verbose)
            res.append(job)
        properties = delayed(_merge_property_slices, pure=False)(properties, chunk_grid, slices, list(str_conds.keys()), res)
    else:
        # Single-process job; don't create child processes
        properties = delayed(_solve_eq_at_conditions, pure=False)(dbf, comps, properties, phase_records,
                                                                       callable_dict, list(str_conds.keys()), verbose)

    # Compute equilibrium values of any additional user-specified properties
    output = output if isinstance(output, (list, tuple, set)) else [output]
    # We already computed these properties so don't recompute them
    output = sorted(set(output) - {'GM', 'MU'})
    for out in output:
        if (out is None) or (len(out) == 0):
            continue
        # TODO: How do we know if a specified property should be per_phase or not?
        # For now, we make a best guess
        if (out == 'degree_of_ordering') or (out == 'DOO'):
            per_phase = True
        else:
            per_phase = False
        delayed(properties.merge, pure=False)(delayed(_eqcalculate, pure=False)(dbf, comps, active_phases,
                                                                                          conditions, out,
                                                                                          data=properties,
                                                                                          per_phase=per_phase,
                                                                                          model=models, **calc_opts),
                                                   inplace=True, compat='equals')
    delayed(properties.attrs.__setitem__, pure=False)('created', datetime.utcnow())
    if scheduler is not None:
        properties = dask.compute(properties, get=scheduler)[0]
    return properties
