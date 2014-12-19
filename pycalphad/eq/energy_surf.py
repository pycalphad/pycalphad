"""
The energy_surf module contains a routine for calculating the
energy surface of a system.
"""

from __future__ import division
from pycalphad import Model
from pycalphad.eq.utils import make_callable, point_sample
import pycalphad.variables as v
import pandas as pd
import numpy as np
try:
    set
except NameError:
    from sets import Set as set #pylint: disable=W0622

def energy_surf(db, comps, phases,
                points_per_phase=10000, ast='numpy', **kwargs):
    """
    Calculate the energy surface of a system containing the specified
    components and phases. Model parameters are taken from 'db' and any
    state variables (T, P, etc.) can be specified as keyword arguments.
    Parameters
    ----------
    db : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names (case-sensitive) of components to consider in the calculation.
    phases : list
        Names (case-sensitive) of phases to consider in the calculation.
    points_per_phase : int, optional
        Approximate number of points to sample per phase.
    ast : ['numpy'], optional
        Specify how we should construct the callable for the energy.

    Returns
    -------
    DataFrame of the energy surface.

    Examples
    --------
    None yet.
    """
    # Here we would check for any keyword arguments that are special, i.e.,
    # there may be keyword arguments that aren't state variables

    # Convert keyword strings to proper state variable objects
    # If we don't do this, sympy will get confused during substitution
    statevars = \
        dict((v.StateVariable(key), value) for (key, value) in kwargs.items())
    active_comps = set(comps)
    # Consider only the active phases
    active_phases = dict((name, db.phases[name]) for name in phases)
    comp_sets = {}
    # Construct a dataframe to hold all the data
    all_phases_df = pd.DataFrame()
    for phase_name, phase_obj in active_phases.items():
        # Build the symbolic representation of the energy
        mod = Model(db, comps, phase_name)
        # Construct an ordered list of the variables
        variables = []
        sublattice_dof = []
        for idx, sublattice in enumerate(phase_obj.constituents):
            dof = 0
            for component in set(sublattice).intersection(active_comps):
                variables.append(v.SiteFraction(phase_name, idx, component))
                dof += 1
            sublattice_dof.append(dof)

        # Build the "fast" representation of that model
        comp_sets[phase_name] = make_callable(mod.ast, \
            list(statevars.keys()) + variables, mode=ast)

        # Make user-friendly site fraction column labels
        var_names = ['Y('+variable.phase_name+',' + \
                str(variable.sublattice_index) + ',' + variable.species +')' \
                for variable in variables]

        # Calculate the number of components in each sublattice
        nontrivial_sublattices = len(sublattice_dof) - sublattice_dof.count(1)
        # Get the site ratios in each sublattice
        site_ratios = list(phase_obj.sublattices)
        # Choose a sensible number of compositions to sample
        num_points = None
        if nontrivial_sublattices > 0:
            num_points = int(points_per_phase**(1/nontrivial_sublattices))
        else:
            # Fixed stoichiometry
            num_points = 1
        # Sample composition space
        points = point_sample(sublattice_dof, size=num_points)
        # Allocate space for energies, once calculated
        energies = np.zeros(len(points))

        # Normalize site ratios
        site_ratio_normalization = 0
        for idx, sublattice in enumerate(phase_obj.constituents):
            # sublattices with only vacancies don't count
            if len(sublattice) == 1 and sublattice[0] == 'VA':
                continue
            site_ratio_normalization += site_ratios[idx]

        site_ratios = [c/site_ratio_normalization for c in site_ratios]

        # TODO: not very efficient point sampling strategy
        # in principle, this could be parallelized
        for idx, point in enumerate(points):
            energies[idx] = \
                comp_sets[phase_name](
                    *(list(statevars.values()) + list(point))
                )

        # Add points and calculated energies to the DataFrame
        data_dict = {'GM':energies, 'Phase':phase_name}
        data_dict.update(kwargs)

        for comp in sorted(comps):
            if comp == 'VA':
                continue
            data_dict['X('+comp+')'] = [0 for n in range(len(points))]

        for column_idx, data in enumerate(points.T):
            data_dict[var_names[column_idx]] = data

        # Now map the internal degrees of freedom to global coordinates
        for p_idx, p in enumerate(points):
            for idx, coordinate in enumerate(p):
                cur_var = variables[idx]
                if cur_var.species == 'VA':
                    continue
                ratio = site_ratios[cur_var.sublattice_index]
                data_dict['X('+cur_var.species+')'][p_idx] += ratio*coordinate

        phase_df = pd.DataFrame(data_dict)
        # Merge dataframe into master dataframe
        # TODO: Better way to do this than concat (always copies)?
        all_phases_df = \
            pd.concat([all_phases_df, phase_df], axis=0, join='outer', \
                        ignore_index=True)
    # all_phases_df now contains energy surface information for the system
    return all_phases_df
