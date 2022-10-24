"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""
import warnings
from collections import Iterable, OrderedDict
from datetime import datetime
from pycalphad.core.workspace import Workspace
from pycalphad.core.light_dataset import LightDataset
import numpy as np
from pycalphad.property_framework import as_property


def equilibrium(dbf, comps, phases, conditions, output=None, model=None,
                verbose=False, calc_opts=None, to_xarray=True,
                parameters=None, solver=None, phase_records=None, **kwargs):
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
    calc_opts : dict, optional
        Keyword arguments to pass to `calculate`, the energy/property calculation routine.
    to_xarray : bool
        Whether to return an xarray Dataset (True, default) or an EquilibriumResult.
    parameters : dict, optional
        Maps SymEngine Symbol to numbers, for overriding the values of parameters in the Database.
    solver : pycalphad.core.solver.SolverBase
        Instance of a solver that is used to calculate local equilibria.
        Defaults to a pycalphad.core.solver.Solver.
    callables : dict, optional
        Pre-computed callable functions for equilibrium calculation.
    phase_records : Optional[Mapping[str, PhaseRecord]]
        Mapping of phase names to PhaseRecord objects with `'GM'` output. Must include
        all active phases. The `model` argument must be a mapping of phase names to
        instances of Model objects.

    Returns
    -------
    Structured equilibrium calculation

    Examples
    --------
    None yet.
    """
    if output is None:
        output = set()
    elif (not isinstance(output, Iterable)) or isinstance(output, str):
        output = [output]
    wks = Workspace(dbf=dbf, comps=comps, phases=phases, conditions=conditions, models=model, parameters=parameters,
                    verbose=verbose, calc_opts=calc_opts, solver=solver, phase_record_factory=phase_records)

    # Compute equilibrium values of any additional user-specified properties
    # We already computed these properties so don't recompute them
    properties = wks.eq
    conds_keys = [str(k) for k in properties.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
    output = sorted(set(output) - {'GM', 'MU'})
    for out in output:
        cprop = as_property(out)
        out = str(cprop)
        result_array = np.zeros(properties.GM.shape) # Will not work for non-scalar properties
        for index, composition_sets in wks.enumerate_composition_sets():
            cur_conds = OrderedDict(zip(conds_keys,
                                        [np.asarray(properties.coords[b][a], dtype=np.float_)
                                        for a, b in zip(index, conds_keys)]))
            chemical_potentials = properties.MU[index]
            result_array[index] = cprop.compute_property(composition_sets, cur_conds, chemical_potentials)
        result = LightDataset({out: (conds_keys, result_array)}, coords=properties.coords)
        properties.merge(result, inplace=True, compat='equals')
    if to_xarray:
        properties = wks.eq.get_dataset()
    properties.attrs['created'] = datetime.utcnow().isoformat()
    if len(kwargs) > 0:
        warnings.warn('The following equilibrium keyword arguments were passed, but unused:\n{}'.format(kwargs))
    return properties
