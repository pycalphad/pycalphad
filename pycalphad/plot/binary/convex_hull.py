from __future__ import print_function
import numpy as np
from collections import OrderedDict
from pycalphad import calculate, variables as v
from pycalphad.core.lower_convex_hull import lower_convex_hull
from pycalphad.core.equilibrium import _adjust_conditions
from pycalphad.core.equilibrium_result import EquilibriumResult

def convex_hull(dbf, comps, phases, conditions, model=None, calc_opts=None, parameters=None, callables=None):
    """
    1D convex hull for fixed potentials.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list or dict
        Names of phases to consider in the calculation.
    conditions : dict
        StateVariables and their corresponding value.
    model : Model, a dict of p  hase names to Model, or a seq of both, optional
        Model class to use for each phase.
    calc_opts : dict, optional
        Keyword arguments to pass to `calculate`, the energy/property calculation routine.
    parameters : dict, optional
        Maps SymPy Symbol to numbers, for overriding the values of parameters in the Database.
    callables : dict, optional
        Pre-computed callable functions for equilibrium calculation.

    Returns
    -------
    tuple
        Tuple of (Gibbs energies, phases, phase fractions, compositions, site fractions, chemical potentials)

    Notes
    -----
    Assumes that potentials are fixed and there is just a 1d composition grid.
    Minimizes the use of Dataset objects.
    """
    from pycalphad import __version__ as pycalphad_version
    calc_opts = calc_opts or {}
    conditions = _adjust_conditions(conditions)

    # 'calculate' accepts conditions through its keyword arguments
    if 'pdens' not in calc_opts:
        calc_opts['pdens'] = 2000
    grid = calculate(dbf, comps, phases, T=conditions[v.T], P=conditions[v.P],
                     parameters=parameters, fake_points=True, output='GM',
                     callables=callables, model=model, N=1, **calc_opts)


    active_phases = sorted(phases)
    # Ensure that '_FAKE_' will fit in the phase name array
    max_phase_name_len = max(max([len(x) for x in active_phases]), 6)
    from pycalphad.core.utils import generate_dof, unpack_components, get_state_variables, instantiate_models, get_pure_elements
    models = instantiate_models(dbf, comps, phases, model=model)
    active_comps = unpack_components(dbf, comps)
    maximum_internal_dof = 0
    for name, ph_obj in dbf.phases.items():
        dof = generate_dof(ph_obj, active_comps)
        maximum_internal_dof = max((len(dof[0]), maximum_internal_dof))

    state_variables = get_state_variables(models=models, conds=conditions)
    nonvacant_elements = get_pure_elements(dbf, comps)
    coord_dict = OrderedDict([(str(key), value) for key, value in conditions.items()])
    coord_dict.update({key: value/10 for key, value in coord_dict.items() if isinstance(key, v.X)})
    grid_shape = tuple(len(x) for x in coord_dict.values())
    coord_dict['vertex'] = np.arange(
        len(nonvacant_elements) + 1)  # +1 is to accommodate the degenerate degree of freedom at the invariant reactions
    coord_dict['component'] = nonvacant_elements
    conds_as_strings = [str(k) for k in conditions.keys()]
    specified_elements = set()
    for i in conditions.keys():
        # Assume that a condition specifying a species contributes to constraining it
        if not hasattr(i, 'species'):
            continue
        specified_elements |= set(i.species.constituents.keys()) - {'VA'}
    dependent_comp = set(nonvacant_elements) - specified_elements
    if len(dependent_comp) != 1:
        raise ValueError('Number of dependent components different from one')
    result = EquilibriumResult({'NP':     (conds_as_strings + ['vertex'], np.empty(grid_shape + (len(nonvacant_elements)+1,))),
                      'GM':     (conds_as_strings, np.empty(grid_shape)),
                      'MU':     (conds_as_strings + ['component'], np.empty(grid_shape + (len(nonvacant_elements),))),
                      'X':      (conds_as_strings + ['vertex', 'component'],
                                 np.empty(grid_shape + (len(nonvacant_elements)+1, len(nonvacant_elements),))),
                      'Y':      (conds_as_strings + ['vertex', 'internal_dof'],
                                 np.empty(grid_shape + (len(nonvacant_elements)+1, maximum_internal_dof,))),
                      'Phase':  (conds_as_strings + ['vertex'],
                                 np.empty(grid_shape + (len(nonvacant_elements)+1,), dtype='U%s' % max_phase_name_len)),
                      'points': (conds_as_strings + ['vertex'],
                                 np.empty(grid_shape + (len(nonvacant_elements)+1,), dtype=np.int32))
                      },
                     coords=coord_dict, attrs={'engine': 'pycalphad %s' % pycalphad_version})
    result = lower_convex_hull(grid, state_variables, result)
    GM_values = result["GM"].squeeze()
    simplex_phases = result["Phase"].squeeze()
    phase_fractions = result["NP"].squeeze()
    phase_compositions = result["X"].squeeze()
    phase_site_fracs = result["Y"].squeeze()
    chempots = result["MU"].squeeze()
    return GM_values, simplex_phases, phase_fractions, phase_compositions, phase_site_fracs, chempots, grid
