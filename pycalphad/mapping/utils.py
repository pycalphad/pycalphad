from __future__ import print_function
from operator import pos, neg
import pycalphad.variables as v
from pycalphad.core.utils import unpack_components, unpack_phases, filter_phases
from pycalphad import calculate, Model
from pycalphad.core.errors import EquilibriumError, ConditionError
from pycalphad.core.lower_convex_hull import lower_convex_hull
from pycalphad.codegen.callables import build_callables
from pycalphad.core.solver import InteriorPointSolver
from pycalphad.core.equilibrium import _adjust_conditions
import dask
from dask import delayed
from xarray import Dataset
from collections import OrderedDict
from datetime import datetime

from .compsets import BinaryCompSet
import numpy as np

def convex_hull(dbf, comps, phases, conditions, output=None, model=None,
                verbose=False, broadcast=True, calc_opts=None,
                scheduler='sync',
                parameters=None, solver=None, callables=None, **kwargs):
    """
    Quickly modified version of `equilibrium` that only calculates the lower convex hull.

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
    parameters : dict, optional
        Maps SymPy Symbol to numbers, for overriding the values of parameters in the Database.
    solver : pycalphad.core.solver.SolverBase
        Instance of a solver that is used to calculate local equilibria.
        Defaults to a pycalphad.core.solver.InteriorPointSolver.
    callables : dict, optional
        Pre-computed callable functions for equilibrium calculation.

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
    comps = sorted(unpack_components(dbf, comps))
    phases = unpack_phases(phases) or sorted(dbf.phases.keys())
    # remove phases that cannot be active
    list_of_possible_phases = filter_phases(dbf, comps)
    active_phases = sorted(set(list_of_possible_phases).intersection(set(phases)))
    if len(list_of_possible_phases) == 0:
        raise ConditionError('There are no phases in the Database that can be active with components {0}'.format(comps))
    if len(active_phases) == 0:
        raise ConditionError('None of the passed phases ({0}) are active. List of possible phases: {1}.'.format(phases, list_of_possible_phases))
    if isinstance(comps, (str, v.Species)):
        comps = [comps]
    if len(set(comps) - set(dbf.species)) > 0:
        raise EquilibriumError('Components not found in database: {}'
                               .format(','.join([c.name for c in (set(comps) - set(dbf.species))])))
    indep_vars = ['T', 'P']
    calc_opts = calc_opts if calc_opts is not None else dict()
    model = model if model is not None else Model
    solver = solver if solver is not None else InteriorPointSolver(verbose=verbose)
    parameters = parameters if parameters is not None else dict()
    if isinstance(parameters, dict):
        parameters = OrderedDict(sorted(parameters.items(), key=str))
    # Modify conditions values to be within numerical limits, e.g., X(AL)=0
    # Also wrap single-valued conditions with lists
    conds = _adjust_conditions(conditions)
    for cond in conds.keys():
        if isinstance(cond, (v.Composition, v.ChemicalPotential)) and cond.species not in comps:
            raise ConditionError('{} refers to non-existent component'.format(cond))
    str_conds = OrderedDict((str(key), value) for key, value in conds.items())
    num_calcs = np.prod([len(i) for i in str_conds.values()])
    components = [x for x in sorted(comps)]
    desired_active_pure_elements = [list(x.constituents.keys()) for x in components]
    desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements for el in constituents]
    pure_elements = sorted(set([x for x in desired_active_pure_elements if x != 'VA']))
    other_output_callables = {}
    if verbose:
        print('Components:', ' '.join([str(x) for x in comps]))
        print('Phases:', end=' ')
    output = output if output is not None else 'GM'
    output = output if isinstance(output, (list, tuple, set)) else [output]
    output = set(output)
    output |= {'GM'}
    output = sorted(output)
    for o in output:
        if o == 'GM':
            eq_callables = build_callables(dbf, comps, active_phases, model=model,
                                           parameters=parameters,
                                           output=o, build_gradients=True, callables=callables,
                                           verbose=verbose)
        else:
            other_output_callables[o] = build_callables(dbf, comps, active_phases, model=model,
                                                        parameters=parameters,
                                                        output=o, build_gradients=False,
                                                        verbose=False)

    phase_records = eq_callables['phase_records']
    models = eq_callables['model']
    maximum_internal_dof = max(len(mod.site_fractions) for mod in models.values())
    if verbose:
        print('[done]', end='\n')

    # 'calculate' accepts conditions through its keyword arguments
    grid_opts = calc_opts.copy()
    grid_opts.update({key: value for key, value in str_conds.items() if key in indep_vars})
    if 'pdens' not in grid_opts:
        grid_opts['pdens'] = 500
    coord_dict = str_conds.copy()
    coord_dict['vertex'] = np.arange(len(pure_elements) + 1)  # +1 is to accommodate the degenerate degree of freedom at the invariant reactions
    grid_shape = np.meshgrid(*coord_dict.values(),
                             indexing='ij', sparse=False)[0].shape
    coord_dict['component'] = pure_elements

    grid = delayed(calculate, pure=False)(dbf, comps, active_phases, output='GM',
                                          model=models, fake_points=True, callables=eq_callables,
                                          parameters=parameters, **grid_opts)

    max_phase_name_len = max(len(name) for name in active_phases)
    # Need to allow for '_FAKE_' psuedo-phase
    max_phase_name_len = max(max_phase_name_len, 6)

    properties = delayed(Dataset, pure=False)({'NP': (list(str_conds.keys()) + ['vertex'],
                                                      np.empty(grid_shape)),
                                               'GM': (list(str_conds.keys()),
                                                      np.empty(grid_shape[:-1])),
                                               'MU': (list(str_conds.keys()) + ['component'],
                                                      np.empty(grid_shape[:-1] + (len(pure_elements),))),
                                               'X': (list(str_conds.keys()) + ['vertex', 'component'],
                                                     np.empty(grid_shape + (len(pure_elements),))),
                                               'Y': (list(str_conds.keys()) + ['vertex', 'internal_dof'],
                                                     np.empty(grid_shape + (maximum_internal_dof,))),
                                               'Phase': (list(str_conds.keys()) + ['vertex'],
                                                         np.empty(grid_shape, dtype='U%s' % max_phase_name_len)),
                                               'points': (list(str_conds.keys()) + ['vertex'],
                                                          np.empty(grid_shape, dtype=np.int32))
                                               },
                                              coords=coord_dict,
                                              attrs={'engine': 'pycalphad %s' % pycalphad_version},
                                              )
    # One last call to ensure 'properties' and 'grid' are consistent with one another
    properties = delayed(lower_convex_hull, pure=False)(grid, properties)
    if scheduler is not None:
        properties = dask.compute(properties, scheduler=scheduler)[0]
    properties.attrs['created'] = datetime.utcnow().isoformat()
    return properties


def get_num_phases(eq_dataset):
    """Return the number of phases in equilibrium from an equilibrium dataset"""
    return int(np.sum(eq_dataset.Phase.values != '', axis=-1, dtype=np.int))

def get_compsets(eq_dataset, indep_comp=None, indep_comp_index=None):
    """Return a list of composition sets in an equilibrium dataset."""
    if indep_comp is None:
        indep_comp = [c for c in eq_dataset.coords if 'X_' in c][0][2:]
    if indep_comp_index is None:
        indep_comp_index = eq_dataset.component.values.tolist().index(indep_comp)
    return BinaryCompSet.from_dataset_vertices(eq_dataset, indep_comp, indep_comp_index, 3)


def close_zero_or_one(val, tol):
    zero = np.isclose(0, val, atol=tol)
    one = np.isclose(1, val, atol=tol)
    return zero or one


def close_to_same(val_1, val_2, tol):
    return np.isclose(val_1, val_2, atol=tol)


def sort_x_by_y(x, y):
    """Sort a list of x in the order of sorting y"""
    return [xx for _, xx in sorted(zip(y, x), key=lambda pair: pair[0])]

def opposite_direction(direction):
    return neg if direction is pos else pos


def find_two_phase_region_compsets(dataset, indep_comp_coord, discrepancy_tol=0.01):
    """
    From a dataset at constant T and P, return the composition sets for a two
    phase region or that have the smallest index composition coordinate

    Parameters
    ----------
    dataset : xr.Dataset
        Equilibrium-like from pycalphad that has a `Phase` Data variable.
    indep_comp_coord : str
        Coordinate name of the independent component

    Returns
    -------
    list
        List of two composition sets for different phases
    """
    for i in range(dataset.sizes[indep_comp_coord]):
        cs = get_compsets(dataset.isel({indep_comp_coord: i}))
        if len(set([c.phase_name for c in cs])) == 2:
            # we found a multiphase region, return them if the discrepancy is
            # above the tolerance
            if cs[0].xdiscrepancy(cs[1], ignore_phase=True) > discrepancy_tol:
                return cs
    return []
