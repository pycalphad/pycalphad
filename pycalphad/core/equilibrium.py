"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""
from __future__ import print_function
import pycalphad.variables as v
from pycalphad.core.utils import unpack_kwarg
from pycalphad.core.utils import sizeof_fmt
from pycalphad.core.utils import unpack_condition, unpack_phases
from pycalphad import calculate, Model
from pycalphad.constraints import mole_fraction
from pycalphad.core.lower_convex_hull import lower_convex_hull
from pycalphad.core.autograd_utils import build_functions
from pycalphad.core.constants import MIN_SITE_FRACTION, COMP_DIFFERENCE_TOL
from sympy import Add, Symbol
from tqdm import tqdm as progressbar
from xarray import Dataset, DataArray
import numpy as np
import scipy.spatial
from collections import defaultdict, namedtuple, OrderedDict
import itertools
import copy
from datetime import datetime
import sys

# Maximum number of global search iterations
MAX_SEARCH_ITERATIONS = 30
# Maximum number of multi-phase solver iterations
MAX_SOLVE_ITERATIONS = 30
# Minimum step norm size to stop solver
MIN_SOLVE_STEP_NORM = 1e-6
# Minimum step multiplier to stop line search
MIN_SOLVE_ALPHA = 1e-6
# Minimum energy (J/mol-atom) difference between iterations before stopping solver
MIN_SOLVE_ENERGY_PROGRESS = 1e-6
# Maximum number of backtracking iterations
MAX_BACKTRACKING = 5
# Maximum number of Newton steps to take
MAX_NEWTON_ITERATIONS = 1
# If the max of the potential difference between iterations is less than
# MIN_SEARCH_PROGRESS J/mol-atom, stop the global search
MIN_SEARCH_PROGRESS = np.inf
# Minimum norm of a Newton direction before it's "zero"
MIN_DIRECTION_NORM = 1e-12
# initial value of 'alpha' in Newton-Raphson procedure
INITIAL_STEP_SIZE = 1.

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

    Parameters
    ----------
    properties : xarray.Dataset
        Equilibrium calculation data. This will be modified!
    multi_index : tuple
        Index into 'properties' of the condition set of interest.

    """
    phases = list(properties['Phase'].values[multi_index])
    # Are there already removed phases?
    if '' in phases:
        num_phases = phases.index('')
    else:
        num_phases = len(phases)
    phases = properties['Phase'].values[multi_index + np.index_exp[:num_phases]]
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
        comp_matrix = properties['X'].values[multi_index + np.index_exp[indices]]
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
            properties['NP'].values[multi_index + np.index_exp[kept_phase]] += \
                properties['NP'].values[multi_index + np.index_exp[redundant]]
            properties['Phase'].values[multi_index + np.index_exp[redundant]] = ''
    # Eliminate any 'fake points' that made it through the convex hull routine
    # These can show up from phases which aren't defined over all of composition space
    properties['NP'].values[np.nonzero(properties['Phase'].values == '_FAKE_')] = np.nan
    properties['Phase'].values[np.nonzero(properties['Phase'].values == '_FAKE_')] = ''
    # Rewrite properties to delete all the nulled out phase entries
    # Then put them at the end
    # That will let us rewrite 'phases' to have only the independent phases
    # And still preserve convenient indexing of 'properties' with phase_idx
    saved_indices = properties['Phase'].values[multi_index] != ''
    saved_indices = np.arange(len(saved_indices))[saved_indices]
    # TODO: Assumes N=1 always
    properties['NP'].values[multi_index + np.index_exp[:len(saved_indices)]] = \
        properties['NP'].values[multi_index + np.index_exp[saved_indices]] / \
        np.sum(properties['NP'].values[multi_index + np.index_exp[saved_indices]])
    properties['NP'].values[multi_index + np.index_exp[len(saved_indices):]] = np.nan
    properties['Phase'].values[multi_index + np.index_exp[:len(saved_indices)]] = \
        properties['Phase'].values[multi_index + np.index_exp[saved_indices]]
    properties['Phase'].values[multi_index + np.index_exp[len(saved_indices):]] = ''
    properties['X'].values[multi_index + np.index_exp[:len(saved_indices), :]] = \
        properties['X'].values[multi_index + np.index_exp[saved_indices, :]]
    properties['X'].values[multi_index + np.index_exp[len(saved_indices):, :]] = np.nan
    properties['Y'].values[multi_index + np.index_exp[:len(saved_indices), :]] = \
        properties['Y'].values[multi_index + np.index_exp[saved_indices, :]]
    properties['Y'].values[multi_index + np.index_exp[len(saved_indices):, :]] = np.nan


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
                         l_multipliers=None, chem_pots=None):
    """
    Compute the constraint vector and constraint Jacobian matrix.
    """
    num_sitefrac_bals = sum([len(dbf.phases[i].sublattices) for i in phases])
    num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
    indep_sum = np.sum([float(val) for i, val in cur_conds.items() if i.startswith('X_')])
    dependent_comp = set(comps) - set([i[2:] for i in cur_conds.keys() if i.startswith('X_')]) - {'VA'}
    dependent_comp = list(dependent_comp)[0]
    mole_fractions = {}
    num_constraints = num_sitefrac_bals + num_mass_bals
    num_vars = len(site_fracs) + len(phases)
    phase_dof = _compute_phase_dof(dbf, comps, phases)
    l_constraints = np.zeros(num_constraints, dtype=np.float)
    if l_multipliers is None:
        l_multipliers = np.zeros(num_constraints, dtype=np.float)
    if chem_pots is not None:
        # Insert chemical potentials as Lagrange multipliers of mass balance constraints
        l_multipliers[sum([len(dbf.phases[i].sublattices) for i in phases]):
        sum([len(dbf.phases[i].sublattices) for i in phases]) + num_mass_bals] = chem_pots
    # Convenience object for caller so it doesn't need to know about the constraint configuration
    chemical_potentials = l_multipliers[sum([len(dbf.phases[i].sublattices) for i in phases]):
        sum([len(dbf.phases[i].sublattices) for i in phases]) + num_mass_bals]

    constraint_jac = np.zeros((num_constraints, num_vars), dtype=np.float)
    # Ordering of constraints by row: sitefrac bal of each phase, then component mass balance
    # Ordering of constraints by column: site fractions of each phase, then phase fractions
    # First: Site fraction balance constraints
    var_idx = 0
    constraint_offset = 0
    for name in phases:
        for idx in range(len(dbf.phases[name].sublattices)):
            active_in_subl = set(dbf.phases[name].constituents[idx]).intersection(comps)
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
        for name, phase_frac in zip(phases, phase_fracs):
            if mole_fractions.get((name, comp), None) is None:
                mole_fractions[(name, comp)] = build_functions(mole_fraction(dbf.phases[name], comps, comp),
                                                               sorted(set(phase_records[name].variables) - {v.T, v.P},
                                                                      key=str))
            comp_obj, comp_grad, comp_hess = mole_fractions[(name, comp)]
            # current phase frac times the comp_grad
            constraint_jac[constraint_offset,
            var_offset:var_offset + phase_dof[phase_idx]] = \
                phase_frac * np.squeeze(comp_grad(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))
            constraint_jac[constraint_offset, len(site_fracs) + phase_idx] += \
                np.squeeze(comp_obj(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))
            l_constraints[constraint_offset] += \
                phase_frac * np.squeeze(comp_obj(*site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))
            # print('L_CONSTRAINTS[{}] += {}'.format(constraint_offset, phase_frac * np.squeeze(comp_obj(*site_fracs[var_offset:var_offset+phase_dof[phase_idx]]))))
            var_offset += phase_dof[phase_idx]
            phase_idx += 1
        if comp != dependent_comp:
            l_constraints[constraint_offset] -= float(cur_conds['X_' + comp])
            # print('L_CONSTRAINTS[{}] -= {}'.format(constraint_offset, float(cur_conds['X_'+comp])))
        else:
            # TODO: Assuming N=1 (fixed for dependent component)
            l_constraints[constraint_offset] -= (1 - indep_sum)
            # print('L_CONSTRAINTS[{}] -= {}'.format(constraint_offset, (1-indep_sum)))
        #l_constraints[constraint_offset] *= -1
        # print('L_CONSTRAINTS[{}] *= -1'.format(constraint_offset))
        constraint_offset += 1
    return l_constraints, constraint_jac, l_multipliers, chemical_potentials, mole_fractions

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
    num_constraints = len(l_constraints)
    gradient_term = np.zeros((num_vars + num_constraints), dtype=np.float)
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
            -phase_frac * np.squeeze(grad_res)[2:]  # Remove P,T grad part
        gradient_term[len(site_fracs) + phase_idx] = -obj_res
        var_offset += phase_dof[phase_idx]
        phase_idx += 1
    gradient_term[:num_vars] += np.dot(constraint_jac.T, l_multipliers)
    gradient_term[num_vars:] = -l_constraints
    return gradient_term

def _build_multiphase_system(dbf, comps, phases, cur_conds, site_fracs, phase_fracs,
                             l_constraints, constraint_jac, l_multipliers, callable_dict, phase_records):
    # Now build objective Hessian and gradient terms
    var_offset = 0
    phase_idx = 0
    phase_dof = _compute_phase_dof(dbf, comps, phases)
    num_vars = len(site_fracs) + len(phases)
    num_constraints = len(l_constraints)
    l_hessian = np.zeros((num_vars + num_constraints, num_vars + num_constraints), dtype=np.float)
    gradient_term = np.zeros((num_vars + num_constraints), dtype=np.float)
    for name, phase_frac in zip(phases, phase_fracs):
        obj = callable_dict[name]
        hess = phase_records[name].hess
        grad = phase_records[name].grad
        obj_res = obj(*itertools.chain([cur_conds['P'], cur_conds['T']],
                                       site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
                      )
        grad_res = grad(*itertools.chain([cur_conds['P'], cur_conds['T']],
                                         site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
                        )
        gradient_term[var_offset:var_offset + phase_dof[phase_idx]] = \
            -phase_frac * np.squeeze(grad_res)[2:]  # Remove P,T grad part
        gradient_term[len(site_fracs) + phase_idx] = -obj_res
        l_hessian[var_offset:var_offset + phase_dof[phase_idx],
        var_offset:var_offset + phase_dof[phase_idx]] = \
            phase_frac * np.squeeze(hess(*itertools.chain([cur_conds['P'], cur_conds['T']],
                                                          site_fracs[var_offset:var_offset + phase_dof[phase_idx]])
                                         ))[2:, 2:]  # Remove P,T hessian part
        # Phase fraction / site fraction cross derivative
        l_hessian[len(site_fracs) + phase_idx, var_offset:var_offset + phase_dof[phase_idx]] = \
            l_hessian[var_offset:var_offset + phase_dof[phase_idx], len(site_fracs) + phase_idx] = \
            (gradient_term[var_offset:var_offset + phase_dof[phase_idx]] / -phase_frac)
        var_offset += phase_dof[phase_idx]
        phase_idx += 1
    # TODO: N=1 fixed by default
    # l_constraints[constraint_offset] -= 1
    # print('L_CONSTRAINTS', l_constraints.astype(np.longfloat))
    # print('CONSTRAINT_JAC', constraint_jac)
    l_hessian[:num_vars, num_vars:] = -constraint_jac.T
    l_hessian[num_vars:, :num_vars] = constraint_jac
    # print('PURE GRADIENT', gradient_term)
    gradient_term[:num_vars] += np.dot(constraint_jac.T, l_multipliers)
    gradient_term[num_vars:] = -l_constraints
    return l_hessian, gradient_term

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
        calcres = calculate(dbf, comps, [phase], output=output, points=points, broadcast=False, **statevars)
        result[output].values[np.nonzero(current_phase_indices)] = calcres[output].values
    if not per_phase:
        result[output] = (result[output] * data['NP']).sum(dim='vertex', skipna=True)
    else:
        result['Phase'] = data['Phase'].copy()
        result['NP'] = data['NP'].copy()
    return result


def equilibrium(dbf, comps, phases, conditions, output=None, model=None,
                verbose=False, pbar=True, broadcast=True, calc_opts=None, return_grids=False, **kwargs):
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
    pbar : bool, optional
        Show a progress bar.
    broadcast : bool
        If True, broadcast conditions against each other. This will compute all combinations.
        If False, each condition should be an equal-length list (or single-valued).
        Disabling broadcasting is useful for calculating equilibrium at selected conditions,
        when those conditions don't comprise a grid.
    calc_opts : dict, optional
        Keyword arguments to pass to `calculate`, the energy/property calculation routine.
    return_grids : bool, optional
        If True, return a tuple of (equilibrium result, grids), where grids is a list of
        Datasets containing the global point set at each hull iteration.
        Mainly useful for teaching and debugging.

    Returns
    -------
    Structured equilibrium calculation.

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
    points_dict = dict()
    maximum_internal_dof = 0
    # Modify conditions values to be within numerical limits, e.g., X(AL)=0
    # Also wrap single-valued conditions with lists
    conds = _adjust_conditions(conditions)
    for cond in conds.keys():
        if isinstance(cond, (v.Composition, v.ChemicalPotential)) and cond.species not in comps:
            raise ConditionError('{} refers to non-existent component'.format(cond))
    str_conds = OrderedDict((str(key), value) for key, value in conds.items())
    indep_vals = list([float(x) for x in np.atleast_1d(val)]
                      for key, val in str_conds.items() if key in indep_vars)
    components = [x for x in sorted(comps) if not x.startswith('VA')]
    # Construct models for each phase; prioritize user models
    models = unpack_kwarg(model, default_arg=Model)
    # for debugging
    if return_grids:
        intermediate_grids = []
    if verbose:
        print('Components:', ' '.join(comps))
        print('Phases:', end=' ')
    for name in progressbar(active_phases, desc='Initialize (1/3)', unit='phase', disable=not pbar):
        mod = models[name]
        if isinstance(mod, type):
            models[name] = mod = mod(dbf, comps, name)
        variables = sorted(mod.energy.atoms(v.StateVariable).union({key for key in conditions.keys() if key in [v.T, v.P]}), key=str)
        site_fracs = sorted(mod.energy.atoms(v.SiteFraction), key=str)
        maximum_internal_dof = max(maximum_internal_dof, len(site_fracs))
        # Extra factor '1e-100...' is to work around an annoying broadcasting bug for zero gradient entries
        #models[name].models['_broadcaster'] = 1e-100 * Mul(*variables) ** 3
        out = models[name].energy
        undefs = list(out.atoms(Symbol) - out.atoms(v.StateVariable))
        for undef in undefs:
            out = out.xreplace({undef: float(0)})
        if (not callable_dict.get(name, False)) or not (grad_callable_dict.get(name, False)) \
            or (not hess_callable_dict.get(name, False)):
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
        plane_obj, plane_grad, plane_hess = build_functions(hyperplane, [v.MU(i) for i in comps if i != 'VA']+site_fracs)
        mass_obj, mass_grad, mass_hess = build_functions(Add(*[mole_fraction(dbf.phases[name], comps, i)
                                                               for i in comps if i != 'VA']), site_fracs)
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
        grid_opts['pdens'] = 100

    coord_dict = str_conds.copy()
    coord_dict['vertex'] = np.arange(len(components))
    grid_shape = np.meshgrid(*coord_dict.values(),
                             indexing='ij', sparse=False)[0].shape
    coord_dict['component'] = components
    if verbose:
        print('Computing initial grid', end=' ')
    # TODO: vectorize this entire calculation over the conditions
    # TODO: Every condition-set should have its own grid
    grid = calculate(dbf, comps, active_phases, output='GM',
                     model=models, callables=callable_dict, fake_points=True, **grid_opts)
    if return_grids:
        intermediate_grids.append(grid)

    if verbose:
        print('[{0} points, {1}]'.format(len(grid.points), sizeof_fmt(grid.nbytes)), end='\n')

    properties = Dataset({'NP': (list(str_conds.keys()) + ['vertex'],
                                 np.empty(grid_shape)),
                          'GM': (list(str_conds.keys()),
                                 np.empty(grid_shape[:-1])),
                          'MU': (list(str_conds.keys()) + ['component'],
                                 np.empty(grid_shape)),
                          'points': (list(str_conds.keys()) + ['vertex'],
                                     np.empty(grid_shape, dtype=np.int))
                          },
                          coords=coord_dict,
                          attrs={'hull_iterations': 1, 'solve_iterations': 0,
                                 'engine': 'pycalphad %s' % pycalphad_version},
                         )
    # Store the potentials from the previous iteration
    current_potentials = properties.MU.copy()

    convex_progress = progressbar(range(MAX_SEARCH_ITERATIONS), desc='Global Search (2/3)', disable=not pbar)
    for iteration in convex_progress:
        if verbose:
            print('Computing convex hull [iteration {}]'.format(properties.attrs['hull_iterations']))
        # lower_convex_hull will modify properties
        lower_convex_hull(grid, properties, verbose=verbose)
        progress = np.abs(current_potentials - properties.MU).values
        converged = (progress < MIN_SEARCH_PROGRESS).all(axis=-1)
        if verbose:
            print('progress', progress.max(), '[{} conditions updated]'.format(np.sum(~converged)))
        if progress.max() < MIN_SEARCH_PROGRESS:
            if verbose:
                print('Global search complete')
            convex_progress.close()
            break
        current_potentials[...] = properties.MU.values
        if verbose:
            print('Refining convex hull')
        # Insert extra dimensions for non-T,P conditions so GM broadcasts correctly
        energy_broadcast_shape = grid.GM.values.shape[:len(indep_vals)] + \
            (1,) * (len(str_conds) - len(indep_vals)) + (grid.GM.values.shape[-1],)
        driving_forces = np.einsum('...i,...i',
                                   properties.MU.values[..., np.newaxis, :].astype(np.float),
                                   grid.X.values[np.index_exp[...] +
                                                 (np.newaxis,) * (len(str_conds) - len(indep_vals)) +
                                                 np.index_exp[:, :]].astype(np.float)) - \
            grid.GM.values.view().reshape(energy_broadcast_shape)

        for name in active_phases:
            dof = len(models[name].energy.atoms(v.SiteFraction))
            current_phase_indices = (grid.Phase.values == name).reshape(energy_broadcast_shape[:-1] + (-1,))
            # Broadcast to capture all conditions
            current_phase_indices = np.broadcast_arrays(current_phase_indices,
                                                        np.empty(driving_forces.shape))[0]
            # This reshape is safe as long as phases have the same number of points at all indep. conditions
            current_phase_driving_forces = driving_forces[current_phase_indices].reshape(
                current_phase_indices.shape[:-1] + (-1,))
            # Note: This works as long as all points are in the same phase order for all T, P
            current_site_fractions = grid.Y.values[..., current_phase_indices[(0,) * len(str_conds)], :]
            if np.sum(current_site_fractions[(0,) * len(indep_vals)][..., :dof]) == dof:
                # All site fractions are 1, aka zero internal degrees of freedom
                # Impossible to refine these points, so skip this phase
                points_dict[name] = current_site_fractions[(0,) * len(indep_vals)][..., :dof]
                continue
            # Find the N points with largest driving force for a given set of conditions
            # Remember that driving force has a sign, so we want the "most positive" values
            # N is the number of components, in this context
            # N points define a 'best simplex' for every set of conditions
            # We also need to restrict ourselves to one phase at a time
            num_kept_points = 2*len(components)
            trial_indices = np.argpartition(current_phase_driving_forces,
                                            -num_kept_points, axis=-1)[..., -num_kept_points:]
            trial_indices = trial_indices.ravel()
            statevar_indices = np.unravel_index(np.arange(np.multiply.reduce(properties.GM.values.shape + (num_kept_points,))),
                                                properties.GM.values.shape + (num_kept_points,))[:len(indep_vals)]
            points = current_site_fractions[np.index_exp[statevar_indices + (trial_indices,)]]
            points.shape = properties.points.shape[:-1] + (-1, maximum_internal_dof)
            # The Y arrays have been padded, so we should slice off the padding
            points = points[..., :dof]
            starting_points = points.copy()
            #print('Starting points shape: ', points.shape)
            #print(points)
            if len(points) == 0:
                if name in points_dict:
                    del points_dict[name]
                # No nearly stable points: skip this phase
                continue

            num_vars = len(phase_records[name].variables)
            plane_grad = phase_records[name].plane_grad
            plane_hess = phase_records[name].plane_hess
            statevar_grid = np.meshgrid(*itertools.chain(indep_vals), sparse=True, indexing='ij')
            # TODO: A more sophisticated treatment of constraints
            num_constraints = len(dbf.phases[name].sublattices)
            constraint_jac = np.zeros((num_constraints, num_vars-len(indep_vars)))
            # Independent variables are always fixed (in this limited implementation)
            #for idx in range(len(indep_vals)):
            #    constraint_jac[idx, idx] = 1
            # This is for site fraction balance constraints
            var_idx = 0#len(indep_vals)
            for idx in range(len(dbf.phases[name].sublattices)):
                active_in_subl = set(dbf.phases[name].constituents[idx]).intersection(comps)
                constraint_jac[idx,
                               var_idx:var_idx + len(active_in_subl)] = 1
                var_idx += len(active_in_subl)

            newton_iteration = 0
            new_grad = None
            new_cast_grad = None
            new_flattened_points = None
            while newton_iteration < MAX_NEWTON_ITERATIONS:
                if new_grad is None:
                    flattened_points = points.reshape(points.shape[:len(indep_vals)] + (-1, points.shape[-1]))
                    grad_args = itertools.chain([i[..., None] for i in statevar_grid],
                                                [flattened_points[..., i] for i in range(flattened_points.shape[-1])])
                    grad = np.array(phase_records[name].grad(*grad_args), dtype=np.float)
                    # Remove derivatives wrt T,P
                    grad = grad[..., len(indep_vars):]
                    grad.shape = points.shape
                else:
                    grad = new_grad
                    flattened_points = new_flattened_points
                grad[np.isnan(grad).any(axis=-1)] = 0  # This is necessary for gradients on the edge of space
                hess_args = itertools.chain([i[..., None] for i in statevar_grid],
                                            [flattened_points[..., i] for i in range(flattened_points.shape[-1])])
                hess = np.array(phase_records[name].hess(*hess_args), dtype=np.float)
                # Remove derivatives wrt T,P
                hess = hess[..., len(indep_vars):, len(indep_vars):]
                hess.shape = points.shape + (hess.shape[-1],)
                hess[np.isnan(hess).any(axis=(-2, -1))] = np.eye(hess.shape[-1])
                plane_args = itertools.chain([properties.MU.values[..., i][..., None] for i in range(properties.MU.shape[-1])],
                                             [points[..., i] for i in range(points.shape[-1])])
                if new_cast_grad is None:
                    cast_grad = np.array(plane_grad(*plane_args), dtype=np.float)
                    # Remove derivatives wrt chemical potentials
                    cast_grad = cast_grad[..., properties.MU.shape[-1]:]
                else:
                    cast_grad = new_cast_grad
                grad = grad - cast_grad
                plane_args = itertools.chain([properties.MU.values[..., i][..., None] for i in range(properties.MU.shape[-1])],
                                             [points[..., i] for i in range(points.shape[-1])])
                cast_hess = np.array(plane_hess(*plane_args), dtype=np.float)
                # Remove derivatives wrt chemical potentials
                cast_hess = cast_hess[..., properties.MU.shape[-1]:, properties.MU.shape[-1]:]
                cast_hess = -cast_hess + hess
                hess = cast_hess.astype(np.float, copy=False)
                try:
                    e_matrix = np.linalg.inv(hess)
                except np.linalg.LinAlgError:
                    print(hess)
                    print(points)
                    raise
                current = calculate(dbf, comps, name, output='GM',
                                    model=models, callables=callable_dict,
                                    fake_points=False,
                                    points=points.reshape(points.shape[:len(indep_vals)] + (-1, points.shape[-1])),
                                    **grid_opts)
                current_plane = np.multiply(current.X.values.reshape(points.shape[:-1] + (len(components),)),
                                            properties.MU.values[..., np.newaxis, :]).sum(axis=-1)
                current_df = current.GM.values.reshape(points.shape[:-1]) - current_plane
                #print('Inv hess check: ', np.isnan(e_matrix).any())
                #print('grad check: ', np.isnan(grad).any())
                dy_unconstrained = -np.einsum('...ij,...j->...i', e_matrix, grad)
                #print('dy_unconstrained check: ', np.isnan(dy_unconstrained).any())
                proj_matrix = np.dot(e_matrix, constraint_jac.T)
                inv_matrix = np.rollaxis(np.dot(constraint_jac, proj_matrix), 0, -1)
                inv_term = np.linalg.inv(inv_matrix)
                #print('inv_term check: ', np.isnan(inv_term).any())
                first_term = np.einsum('...ij,...jk->...ik', proj_matrix, inv_term)
                #print('first_term check: ', np.isnan(first_term).any())
                # Normally a term for the residual here
                # We only choose starting points which obey the constraints, so r = 0
                cons_summation = np.einsum('...i,...ji->...j', dy_unconstrained, constraint_jac)
                #print('cons_summation check: ', np.isnan(cons_summation).any())
                cons_correction = np.einsum('...ij,...j->...i', first_term, cons_summation)
                #print('cons_correction check: ', np.isnan(cons_correction).any())
                dy_constrained = dy_unconstrained - cons_correction
                #print('dy_constrained check: ', np.isnan(dy_constrained).any())
                # TODO: Support for adaptive changing independent variable steps
                new_direction = dy_constrained
                #print('new_direction', new_direction)
                #print('points', points)
                # Backtracking line search
                if np.isnan(new_direction).any():
                    print('new_direction', new_direction)
                #print('Convergence angle:', -(grad*new_direction).sum(axis=-1) / (np.linalg.norm(grad, axis=-1) * np.linalg.norm(new_direction, axis=-1)))
                new_points = points + INITIAL_STEP_SIZE * new_direction
                alpha = np.full(new_points.shape[:-1], INITIAL_STEP_SIZE, dtype=np.float)
                #alpha[np.all(np.linalg.norm(new_direction, axis=-1) < MIN_DIRECTION_NORM, axis=-1)] = 0
                negative_points = np.any(new_points < 0., axis=-1)
                while np.any(negative_points):
                    alpha[negative_points] *= 0.5
                    new_points = points + alpha[..., np.newaxis] * new_direction
                    negative_points = np.any(new_points < 0., axis=-1)
                # Backtracking line search
                # alpha now contains maximum possible values that keep us inside the space
                # but we don't just want to take the biggest step; we want the biggest step which reduces energy
                new_points = new_points.reshape(new_points.shape[:len(indep_vals)] + (-1, new_points.shape[-1]))
                candidates = calculate(dbf, comps, name, output='GM',
                                       model=models, callables=callable_dict,
                                       fake_points=False, points=new_points, **grid_opts)
                candidate_plane = np.multiply(candidates.X.values.reshape(points.shape[:-1] + (len(components),)),
                                              properties.MU.values[..., np.newaxis, :]).sum(axis=-1)
                energy_diff = (candidates.GM.values.reshape(new_direction.shape[:-1]) - candidate_plane) - current_df
                new_flattened_points = new_points.reshape(new_points.shape[:len(indep_vals)] + (-1, new_points.shape[-1]))
                new_grad_args = itertools.chain([i[..., None] for i in statevar_grid],
                                            [new_flattened_points[..., i] for i in range(new_flattened_points.shape[-1])])
                new_grad = np.array(phase_records[name].grad(*new_grad_args), dtype=np.float)
                # Remove derivatives wrt T,P
                new_grad = new_grad[..., len(indep_vars):]
                new_points.shape = new_direction.shape
                new_grad.shape = new_points.shape
                new_plane_args = itertools.chain([properties.MU.values[..., i][..., None] for i in range(properties.MU.shape[-1])],
                                             [new_points[..., i] for i in range(new_points.shape[-1])])
                new_cast_grad = np.array(plane_grad(*new_plane_args), dtype=np.float)
                # Remove derivatives wrt chemical potentials
                new_cast_grad = new_cast_grad[..., properties.MU.shape[-1]:]
                new_grad = new_grad - new_cast_grad
                bad_steps = energy_diff > alpha * 1e-4 * (new_direction * grad).sum(axis=-1)
                bad_steps |= np.abs(np.multiply(new_direction, new_grad).sum(axis=-1)) > 0.9*np.abs(np.multiply(new_direction, grad).sum(axis=-1))
                backtracking_iterations = 0
                while np.any(bad_steps):
                    alpha[bad_steps] *= 0.9
                    new_points = points + alpha[..., np.newaxis] * new_direction
                    #print('new_points', new_points)
                    #print('bad_steps', bad_steps)
                    new_points = new_points.reshape(new_points.shape[:len(indep_vals)] + (-1, new_points.shape[-1]))
                    candidates = calculate(dbf, comps, name, output='GM',
                                           model=models, callables=callable_dict,
                                           fake_points=False, points=new_points, **grid_opts)
                    candidate_plane = np.multiply(candidates.X.values.reshape(points.shape[:-1] + (len(components),)),
                                                  properties.MU.values[..., np.newaxis, :]).sum(axis=-1)
                    energy_diff = (candidates.GM.values.reshape(new_direction.shape[:-1]) - candidate_plane) - current_df
                    new_flattened_points = new_points.reshape(new_points.shape[:len(indep_vals)] + (-1, new_points.shape[-1]))
                    new_grad_args = itertools.chain([i[..., None] for i in statevar_grid],
                                                [new_flattened_points[..., i] for i in range(new_flattened_points.shape[-1])])
                    new_grad = np.array(phase_records[name].grad(*new_grad_args), dtype=np.float)
                    # Remove derivatives wrt T,P
                    new_grad = new_grad[..., len(indep_vars):]
                    new_points.shape = new_direction.shape
                    new_grad.shape = new_points.shape
                    new_plane_args = itertools.chain([properties.MU.values[..., i][..., None] for i in range(properties.MU.shape[-1])],
                                                 [new_points[..., i] for i in range(new_points.shape[-1])])
                    new_cast_grad = np.array(plane_grad(*new_plane_args), dtype=np.float)
                    # Remove derivatives wrt chemical potentials
                    new_cast_grad = new_cast_grad[..., properties.MU.shape[-1]:]
                    new_grad = new_grad - new_cast_grad
                    bad_steps = energy_diff > alpha * 1e-4 * (new_direction * grad).sum(axis=-1)
                    bad_directions = (new_direction * grad).sum(axis=-1) > 0
                    #if np.any(bad_directions):
                    #    print('bad points', new_points[np.nonzero(bad_directions)])
                    #    print('bad directions', new_direction[np.nonzero(bad_directions)])
                    #    print('bad gradients', grad[np.nonzero(bad_directions)])
                    #    print('energy_diff', energy_diff[np.nonzero(bad_directions)])
                    bad_steps |= np.abs(np.multiply(new_direction, new_grad).sum(axis=-1)) > 0.9*np.abs(np.multiply(new_direction, grad).sum(axis=-1))
                    backtracking_iterations += 1
                    if backtracking_iterations > MAX_BACKTRACKING:
                        # Couldn't find a suitable step -- take no step for bad steps
                        print('Backtracking failed')
                        new_grad[np.nonzero(~bad_steps)] = grad[np.nonzero(~bad_steps)]
                        new_points[np.nonzero(~bad_steps)] = points[np.nonzero(~bad_steps)]
                        break
                    alpha[alpha < MIN_SOLVE_ALPHA] = 0
                #print('backtracking iterations', backtracking_iterations)
                biggest_step = np.max(np.linalg.norm(new_points - points, axis=-1))
                if biggest_step < MIN_DIRECTION_NORM:
                    if verbose:
                        print('N-R convergence on mini-iteration', newton_iteration, '[{}]'.format(name))
                    points = new_points
                    break
                if verbose:
                    #print('Biggest step:', biggest_step)
                    #print('points', points)
                    #print('grad of points', grad)
                    #print('new_direction', new_direction)
                    #print('alpha', alpha)
                    #print('new_points', new_points)
                    pass
                points = new_points
                newton_iteration += 1
            new_points = points.reshape(points.shape[:len(indep_vals)] + (-1, points.shape[-1]))
            print('points.shape', points.shape)
            print('starting_points.shape', starting_points.shape)
            starting_points.shape = points.shape[:len(indep_vals)] + (-1, points.shape[-1])
            new_points = np.concatenate((starting_points, new_points), axis=-2)
            points_dict[name] = new_points

        if verbose:
            print('Rebuilding grid', end=' ')
        grid = calculate(dbf, comps, active_phases, output='GM',
                         model=models, callables=callable_dict,
                         fake_points=True, points=points_dict, **grid_opts)
        if return_grids:
            intermediate_grids.append(grid)
        if verbose:
            print('[{0} points, {1}]'.format(len(grid.points), sizeof_fmt(grid.nbytes)), end='\n')
        properties.attrs['hull_iterations'] += 1
    # Make sure all the verbose output appears to the user
    if verbose:
        print('Refining equilibrium')
        sys.stdout.flush()
    # One last call to ensure 'properties' and 'grid' are consistent with one another
    lower_convex_hull(grid, properties, verbose=verbose)
    indexer = []
    for idx, vals in enumerate(indep_vals):
        indexer.append(np.arange(len(vals), dtype=np.int)[idx * (np.newaxis,) + np.index_exp[:] + \
                                                          (len(conds.keys())-idx+1) * (np.newaxis,)])
    indexer.append(properties['points'].values[..., np.newaxis])
    indexer.append(np.arange(grid['X'].values.shape[-1], dtype=np.int)[(len(conds.keys())) * (np.newaxis,) + np.index_exp[:]])
    ravelled_X_view = grid['X'].values[tuple(indexer)]
    indexer[-1] = np.arange(grid['Y'].values.shape[-1], dtype=np.int)[(len(conds.keys())) * (np.newaxis,) + np.index_exp[:]]
    ravelled_Y_view = grid['Y'].values[tuple(indexer)]
    indexer = []
    for idx, vals in enumerate(indep_vals):
        indexer.append(np.arange(len(vals), dtype=np.int)[idx * (np.newaxis,) + np.index_exp[:] + \
                                                          (len(conds.keys())-idx) * (np.newaxis,)])
    indexer.append(properties['points'].values)
    ravelled_Phase_view = grid['Phase'].values[tuple(indexer)]
    # Copy final point values from the grid and drop the index array
    # For some reason direct construction doesn't work. We have to create empty and then assign.
    properties['X'] = DataArray(np.empty_like(ravelled_X_view),
                                dims=properties['points'].dims + ('component',))
    properties['X'].values[...] = ravelled_X_view
    properties['Y'] = DataArray(np.empty_like(ravelled_Y_view),
                                dims=properties['points'].dims + ('internal_dof',))
    properties['Y'].values[...] = ravelled_Y_view
    # TODO: What about invariant reactions? We should perform a final driving force calculation here.
    # We can handle that in the same post-processing step where we identify single-phase regions.
    properties['Phase'] = DataArray(np.empty_like(ravelled_Phase_view),
                                    dims=properties['points'].dims)
    properties['Phase'].values[...] = ravelled_Phase_view
    del properties['points']
    it = np.nditer(properties['GM'].values, flags=['multi_index'])
    num_conds = np.prod([len(x) for x in properties['GM'].coords.values()])
    # For single-condition calculations, don't generate a separate progress bar
    multi_phase_progress = progressbar(desc='Solve (3/3)', total=num_conds,
                                       unit='cond', disable=(num_conds==1) or (not pbar))
    while not it.finished:
        # A lot of this code relies on cur_conds being ordered!
        cur_conds = OrderedDict(zip(properties['GM'].coords.keys(),
                                    [b[a] for a, b in zip(it.multi_index, properties['GM'].coords.values())]))
        # sum of independently specified components
        indep_sum = np.sum([float(val) for i, val in cur_conds.items() if i.startswith('X_')])
        if indep_sum > 1:
            # Sum of independent component mole fractions greater than one
            # Skip this condition set
            # We silently allow this to make 2-D composition mapping easier
            properties['MU'].values[it.multi_index] = np.nan
            properties['NP'].values[it.multi_index + np.index_exp[:len(phases)]] = np.nan
            properties['X'].values[it.multi_index + np.index_exp[:len(phases)]] = np.nan
            properties['Y'].values[it.multi_index] = np.nan
            properties['GM'].values[it.multi_index] = np.nan
            it.iternext()
            continue
        dependent_comp = set(comps) - set([i[2:] for i in cur_conds.keys() if i.startswith('X_')]) - {'VA'}
        if len(dependent_comp) == 1:
            dependent_comp = list(dependent_comp)[0]
        else:
            multi_phase_progress.close()
            raise ValueError('Number of dependent components different from one')
        #chem_pots = OrderedDict(zip(properties.coords['component'].values, properties['MU'].values[it.multi_index]))
        # For single-condition calculations, make a progress bar for the individual solve
        solve_progress = progressbar(range(MAX_SOLVE_ITERATIONS), desc='Solve (3/3',
                                     disable=(num_conds>1) or (not pbar))
        for cur_iter in solve_progress:
            #print('CUR_ITER:', cur_iter)
            phases = list(properties['Phase'].values[it.multi_index])
            if '' in phases:
                old_phase_length = phases.index('')
            else:
                old_phase_length = -1
            remove_degenerate_phases(properties, it.multi_index)
            # TODO: This is a temporary workaround to allow phase diagram calculation to work decently
            if kwargs.get('_approx', None) is not None:
                break
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
            zero_dof = np.all((properties['Y'].values[it.multi_index] == 1.) | np.isnan(properties['Y'].values[it.multi_index]))
            if (num_phases == 1) and zero_dof:
                # Single phase with zero internal degrees of freedom, can't do any refinement
                # TODO: In the future we may be able to refine other degrees of freedom like temperature
                # Chemical potentials have no meaning for this case
                properties['MU'].values[it.multi_index] = np.nan
                break
            phases = properties['Phase'].values[it.multi_index + np.index_exp[:num_phases]]
            #num_sitefrac_bals = sum([len(dbf.phases[i].sublattices) for i in phases])
            #num_mass_bals = len([i for i in cur_conds.keys() if i.startswith('X_')]) + 1
            phase_fracs = properties['NP'].values[it.multi_index + np.index_exp[:len(phases)]]
            phase_dof = [len(set(phase_records[name].variables)-{v.T, v.P}) for name in phases]
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
                        np.sum(site_fracs[var_idx:var_idx +len(active_in_subl)], keepdims=True)
                    var_idx += len(active_in_subl)

            l_constraints, constraint_jac, l_multipliers, old_chem_pots, mole_fraction_funcs = \
                _compute_constraints(dbf, comps, phases, cur_conds, site_fracs, phase_fracs, phase_records,
                                     chem_pots=properties['MU'].values[it.multi_index].copy())
            num_vars = len(site_fracs) + len(phases)
            l_hessian, gradient_term = _build_multiphase_system(dbf, comps, phases, cur_conds, site_fracs, phase_fracs,
                                                                l_constraints, constraint_jac, l_multipliers,
                                                                callable_dict, phase_records)
            try:
                step = np.linalg.solve(l_hessian, gradient_term)
            except np.linalg.LinAlgError:
                print('Failed to compute ', cur_conds)
                properties['GM'].values[it.multi_index] = np.nan
                solve_progress.close()
                break
            if np.any(np.isnan(step)):
                print('PHASES: ', phases)
                print('SITE FRACTIONS: ', site_fracs)
                print('PHASE FRACTIONS: ', phase_fracs)
                print('HESSIAN: ', l_hessian)
                solve_progress.close()
                print('Bad step: '+str(step))
                break
            # Backtracking line search
            # First restrict alpha to steps in the feasible region
            alpha = 1
            #while ((np.any((site_fracs + alpha * step[:len(site_fracs)]) < 0.1*MIN_SITE_FRACTION) or
            #       np.any((phase_fracs + alpha * step[len(site_fracs):len(site_fracs)+len(phases)]) < 0)) and
            #       alpha > MIN_SOLVE_ALPHA):
            #    alpha *= 0.999
            #if alpha <= MIN_SOLVE_ALPHA:
            #    alpha = 0
            #print('INITIAL ALPHA', alpha)
            #print('STEP', step)
            #print('SITE FRACS', site_fracs)
            #print('PHASE FRACS', phase_fracs)
            # Take the largest step which reduces the energy
            old_energy = copy.deepcopy(properties['GM'].values[it.multi_index])
            #print('OLD ENERGY', old_energy)
            # XXX: Why is it necessary to square this?
            old_constrained_objective = old_energy + ((l_multipliers * l_constraints).sum()) ** 2
            #print('OLD OBJ', old_constrained_objective)
            old_chem_pots = properties['MU'].values[it.multi_index].copy()
            #print('STARTING ALPHA', alpha)
            wolfe_conditions = False
            while alpha > MIN_SOLVE_ALPHA:
                #print('ALPHA', alpha)
                candidate_site_fracs = site_fracs + alpha * step[:len(site_fracs)]
                candidate_site_fracs[candidate_site_fracs < MIN_SITE_FRACTION] = MIN_SITE_FRACTION
                candidate_site_fracs[candidate_site_fracs > 1] = 1
                candidate_l_multipliers = l_multipliers + alpha * step[num_vars:]
                #print('CANDIDATE L MULTIPLIERS', candidate_l_multipliers)
                candidate_phase_fracs = phase_fracs + \
                                       alpha * step[len(candidate_site_fracs):len(candidate_site_fracs)+len(phases)]
                candidate_phase_fracs[candidate_phase_fracs < MIN_SITE_FRACTION] = MIN_SITE_FRACTION
                candidate_phase_fracs[candidate_phase_fracs > 1] = 1
                #if len(phases) == len([c for c in comps if c != 'VA']):
                #    # We have the maximum number of phases
                #    # Compute the chemical potentials exactly from the tangent hyperplane
                #    phase_compositions = np.zeros((len(phases), len([c for c in comps if c != 'VA'])))
                #    phase_energies = np.zeros((len(phases)))
                #    var_offset = 0
                #    for phase_idx in range(len(phases)):
                #        for comp_idx, comp in enumerate([c for c in comps if c != 'VA']):
                #            phase_compositions[phase_idx, comp_idx] = \
                #                mole_fraction_funcs[(phases[phase_idx], comp)][0](*candidate_site_fracs[var_offset:var_offset+phase_dof[phase_idx]])
                #        phase_energies[phase_idx] = callable_dict[phases[phase_idx]](
                #            *itertools.chain([cur_conds['P'], cur_conds['T']],
                #                             candidate_site_fracs[var_offset:var_offset + phase_dof[phase_idx]]))
                #        var_offset += phase_dof[phase_idx]
                #    exact_chempots = np.linalg.solve(phase_compositions, phase_energies)
                #else:
                #    exact_chempots = None
                (candidate_l_constraints, candidate_constraint_jac, candidate_l_multipliers, candidate_chem_pots,
                mole_fraction_funcs) = \
                    _compute_constraints(dbf, comps, phases, cur_conds,
                                         candidate_site_fracs, candidate_phase_fracs, phase_records,
                                         l_multipliers=candidate_l_multipliers)
                #print('CANDIDATE L MULTIPLIERS AFTER', candidate_l_multipliers)
                #print('CANDIDATE_L_CONSTRAINTS', candidate_l_constraints)
                #print('CANDIDATE L MULS', candidate_l_constraints * candidate_l_multipliers)
                #print('CANDIDATE L MUL SUM', (candidate_l_multipliers *
                #                                    candidate_l_constraints).sum()
                #      )
                candidate_energy = _compute_multiphase_objective(dbf, comps, phases, cur_conds, candidate_site_fracs,
                                                                 candidate_phase_fracs,
                                                                 callable_dict)
                # XXX: Why is it necessary to square this?
                candidate_constrained_objective = candidate_energy + ((candidate_l_multipliers *
                                                                       candidate_l_constraints).sum()) ** 2
                #print('CANDIDATE CHEM POTS', candidate_chem_pots)
                #print('CANDIDATE ENERGY', candidate_energy)
                #print('CANDIDATE OBJ', candidate_constrained_objective)
                #print('STEP', step[:num_vars])
                #print('GRADIENT TERM', gradient_term)
                #print('CANDIDATE GRADIENT', candidate_gradient_term)

                # Remember that 'gradient_term' is actually the NEGATIVE of the gradient
                wolfe_conditions = (candidate_constrained_objective - old_constrained_objective) <= \
                                   alpha * 1e-4 * (step * -gradient_term).sum(axis=-1)
                #print('WOLFE CONDITION 1', wolfe_conditions)
                # Optimization to avoid costly gradient calculation if Wolfe conditions won't be met anyway
                if wolfe_conditions:
                    candidate_gradient_term = _build_multiphase_gradient(dbf, comps, phases,
                                                                         cur_conds, candidate_site_fracs,
                                                                         candidate_phase_fracs,
                                                                         l_constraints, constraint_jac,
                                                                         l_multipliers, callable_dict, phase_records)
                    #print('CANDIDATE GRAD SUM', np.multiply(step[:num_vars], candidate_gradient_term[:num_vars]).sum())
                    #print('OLD GRAD SUM', np.multiply(step[:num_vars], gradient_term[:num_vars]).sum())
                    wolfe_conditions &= np.abs(np.multiply(step, -candidate_gradient_term).sum(axis=-1)) <= \
                                        0.9*np.abs(np.multiply(step, -gradient_term).sum(axis=-1))
                # Seems to be necessary for some unit tests to explicitly allow chemical potential updates
                chempot_update = (candidate_constrained_objective - old_constrained_objective) <= MIN_SOLVE_ENERGY_PROGRESS
                chempot_update &= np.abs(candidate_chem_pots - old_chem_pots).max() > 0.1
                wolfe_conditions |= chempot_update
                #print('WOLFE CONDITION 1&2', wolfe_conditions)
                if wolfe_conditions:
                    break
                alpha *= 0.5
            #print('RESULT ALPHA', alpha)
            #print('WOLFE CONDITIONS', wolfe_conditions)
            if wolfe_conditions:
                # We updated degrees of freedom this iteration
                properties['MU'].values[it.multi_index] = candidate_chem_pots
                properties['NP'].values[it.multi_index + np.index_exp[:len(phases)]] = candidate_phase_fracs
                properties['X'].values[it.multi_index + np.index_exp[:len(phases)]] = 0
                properties['GM'].values[it.multi_index] = candidate_energy
                var_offset = 0
                for phase_idx in range(len(phases)):
                    properties['Y'].values[it.multi_index + np.index_exp[phase_idx, :phase_dof[phase_idx]]] = \
                        candidate_site_fracs[var_offset:var_offset+phase_dof[phase_idx]]
                    for comp_idx, comp in enumerate([c for c in comps if c != 'VA']):
                        properties['X'].values[it.multi_index + np.index_exp[phase_idx, comp_idx]] = \
                            mole_fraction_funcs[(phases[phase_idx], comp)][0](*candidate_site_fracs[var_offset:var_offset+phase_dof[phase_idx]])
                    var_offset += phase_dof[phase_idx]

            properties.attrs['solve_iterations'] += 1
            if verbose:
                print('Chem pot progress', np.linalg.norm(properties['MU'].values[it.multi_index] - old_chem_pots, ord=np.inf))
                print('Energy progress', properties['GM'].values[it.multi_index] - old_energy)
            no_progress = ~wolfe_conditions
            #no_progress = np.abs(properties['MU'].values[it.multi_index] - old_chem_pots).max() < 0.1
            #no_progress &= np.abs(properties['GM'].values[it.multi_index] - old_energy) < MIN_SOLVE_ENERGY_PROGRESS
            if no_progress:
                if verbose:
                    print('No progress')
                solve_progress.close()
                break
        multi_phase_progress.update()
        it.iternext()
    multi_phase_progress.close()

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
        properties.merge(_eqcalculate(dbf, comps, active_phases, conditions, out,
                                      data=properties, per_phase=per_phase, **calc_opts), inplace=True, compat='equals')
    properties.attrs['created'] = datetime.utcnow()
    if return_grids:
        return properties, intermediate_grids
    else:
        return properties
