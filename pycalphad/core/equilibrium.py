"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from datetime import datetime
import pycalphad.variables as v
from pycalphad.core.workspace import Workspace
from pycalphad.core.light_dataset import LightDataset
from pycalphad.core.solver import Solver
from pycalphad.core.starting_point import starting_point
from pycalphad.core.eqsolver import _solve_eq_at_conditions
from pycalphad.core.utils import filter_phases, instantiate_models
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.property_framework.units import as_quantity
from pycalphad.core.calculate import calculate
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
    wks = Workspace(database=dbf, components=comps, phases=phases, conditions=conditions, models=model, parameters=parameters,
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
                                        [np.asarray(properties.coords[b][a], dtype=np.float64)
                                        for a, b in zip(index, conds_keys)]))
            chemical_potentials = properties.MU[index]
            result_array[index] = cprop.compute_property(composition_sets, cur_conds, chemical_potentials)
        result = LightDataset({out: (conds_keys, result_array)}, coords=properties.coords)
        properties.merge(result, inplace=True, compat='equals')
    if to_xarray:
        properties = wks.eq.get_dataset()
    properties.attrs['created'] = datetime.now().isoformat()
    if len(kwargs) > 0:
        warnings.warn('The following equilibrium keyword arguments were passed, but unused:\n{}'.format(kwargs))
    return properties


def zip_equilibrium(dbf, comps, phases, xs, solutes, T, P, pdens=60, calc_opts=None,
                    solver=None, verbose=False):
    """
    Calculate equilibria at paired composition rows without expanding composition axes.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database.
    comps : list
        Names of components to consider in the calculation.
    phases : list or dict
        Names of phases to consider in the calculation.
    xs : array-like, shape (n_points, n_components_without_va)
        Composition rows. Column 0 is the dependent component. Columns 1 and
        onward correspond to ``solutes``.
    solutes : list
        Independent components matching columns 1 and onward in ``xs``.
    T : float or array-like
        Scalar, paired, or broadcast temperatures in K. A scalar temperature
        returns one result per composition row. A temperature array with the
        same length as ``xs`` pairs each temperature with the corresponding row.
        Any other temperature array is broadcast as temperature-major results.
    P : float
        Pressure in Pa.
    pdens : int
        Point density passed to ``calculate``.
    calc_opts : dict, optional
        Keyword arguments to pass to ``calculate``.
    solver : SolverBase, optional
        Solver instance.
    verbose : bool, optional
        Print details of calculations.

    Returns
    -------
    list of xarray.Dataset
        Equilibrium results for the paired or broadcast points.
    """
    if calc_opts is None:
        calc_opts = {}
    calc_opts = dict(calc_opts)
    calc_opts.setdefault('pdens', pdens)
    if solver is None:
        solver = Solver(verbose=verbose)

    xs = np.asarray(xs, dtype=float)
    temperatures = np.atleast_1d(np.asarray(T, dtype=float))
    point_count = len(xs)

    if len(temperatures) == 1:
        mode = 'scalar'
    elif len(temperatures) == point_count:
        mode = 'paired'
    else:
        mode = 'broadcast'

    if isinstance(phases, str):
        phases = [phases]
    active_phases = [str(phase) for phase in filter_phases(dbf, v.unpack_components(comps), phases)]
    models = instantiate_models(dbf, comps, active_phases)
    state_variables = [v.N, v.P, v.T]
    phase_record_factory = PhaseRecordFactory(dbf, comps, state_variables, models)

    def build_grid(temperature):
        return calculate(dbf, comps, active_phases, model=models, fake_points=True,
                         phase_records=phase_record_factory, output='GM', to_xarray=False,
                         N=1, T=float(temperature), P=P, **calc_opts)

    def solve_row(row, temperature, grid):
        conditions = OrderedDict()
        conditions[v.N] = [1.0]
        conditions[v.T] = [float(temperature)]
        conditions[v.P] = [float(P)]
        for index, solute in enumerate(solutes):
            mole_fraction = float(row[index + 1])
            mole_fraction = min(max(mole_fraction, 1e-10), 1 - 1e-10)
            conditions[v.X(solute)] = [mole_fraction]

        unitless_conditions = OrderedDict(
            (key, as_quantity(key, value).to(key.implementation_units).magnitude)
            for key, value in conditions.items()
        )
        properties = starting_point(unitless_conditions, state_variables, phase_record_factory, grid)
        properties = _solve_eq_at_conditions(properties, phase_record_factory, grid,
                                             list(unitless_conditions.keys()), state_variables,
                                             verbose, solver=solver)
        return properties.get_dataset()

    if mode == 'scalar':
        grid = build_grid(temperatures[0])
        return [solve_row(row, temperatures[0], grid) for row in xs]

    if mode == 'paired':
        groups = defaultdict(list)
        for index, (row, temperature) in enumerate(zip(xs, temperatures)):
            groups[temperature].append((index, row))
        results = [None] * point_count
        for temperature in np.unique(temperatures):
            grid = build_grid(temperature)
            for original_index, row in groups[temperature]:
                results[original_index] = solve_row(row, temperature, grid)
        return results

    results = []
    for temperature in temperatures:
        grid = build_grid(temperature)
        for row in xs:
            results.append(solve_row(row, temperature, grid))
    return results
