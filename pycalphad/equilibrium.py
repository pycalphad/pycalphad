"""
The equilibrium module defines routines for interacting with
calculated phase equilibria.
"""

from tinydb import TinyDB
from tinydb.storages import MemoryStorage
import pycalphad.variables as v
from pycalphad.minimize import point_sample, make_callable
from pycalphad import Model
import pandas as pd
import numpy as np
import scipy.spatial
try:
    set
except NameError:
    from sets import Set as set #pylint: disable=W0622

class Equilibrium(object):
    """
    Calculate the equilibrium state of a system containing the specified
    components and phases, under the specified conditions.
    Model parameters are taken from 'db' and any state variables (T, P, etc.)
    can be specified as keyword arguments.

    Parameters
    ----------
    dbf : Database
        Thermodynamic database containing the relevant parameters.
    comps : list
        Names (case-sensitive) of components to consider in the calculation.
    phases : list
        Names (case-sensitive) of phases to consider in the calculation.
    points_per_phase : int, optional
        Approximate number of points to sample per phase.
    ast : ['theano', 'numpy'], optional
        Specify how we should construct the callable for the energy.

    Returns
    -------
    Structured equilibrium calculation.

    Examples
    --------
    None yet.
    """
    def __init__(self, dbf, comps, phases,
                 points_per_phase=10000, ast='numpy', **kwargs):
        self.conditions = TinyDB(storage=MemoryStorage)
        self.components = set(comps)
        self.phases = dict()
        self.statevars = dict()
        self.data = pd.DataFrame()

        self._phase_callables = dict()
        self._variables = []
        self._varnames = []
        self._sublattice_dof = []
        # Here we would check for any keyword arguments that are special, i.e.,
        # there may be keyword arguments that aren't state variables

        # Convert keyword strings to proper state variable objects
        # If we don't do this, sympy will get confused during substitution
        # TODO: Needs to differentiate T,P from N when specifying conditions
        self.statevars = \
            dict((v.StateVariable(key), value) \
                for (key, value) in kwargs.items())
        # Consider only the active phases
        self.phases = dict((name, dbf.phases[name]) for name in phases)

        self._build_objective_functions(dbf, ast)

        for phase_obj in self.phases.values():
            data_dict = \
                self._calculate_energy_surface(phase_obj, points_per_phase)

            phase_df = pd.DataFrame(data_dict)
            # TODO: Apply phase-specific conditions
            # Merge dataframe into master dataframe
            self.data = \
                pd.concat([self.data, phase_df], axis=0, join='outer', \
                            ignore_index=True)
        # self.data now contains energy surface information for the system
        # determine column indices for independent degrees of freedom
        independent_dof = list(self.components)[:-1]
        for dof_idx, dof_ent in enumerate(independent_dof):
            independent_dof[dof_idx] = 'X(' + dof_ent + ')'
        independent_dof.append('GM')
        # calculate the convex hull for the independent d.o.f
        # TODO: Apply activity conditions here
        hull = scipy.spatial.ConvexHull(
            self.data[independent_dof].values
        )
        # locate the simplex closest to our desired condition
        #candidate_simplex = scipy.spatial.tsearch(hull.simplices, xi)
        # use simplex values as a starting point; refine with optimization

    def _calculate_energy_surface(self, phase_obj, points_per_phase):
        "Sample the energy surface of a phase."
        # Calculate the number of components in each sublattice
        nontrivial_sublattices = \
            len(self._sublattice_dof) - self._sublattice_dof.count(1)
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
        points = point_sample(self._sublattice_dof, size=num_points)
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
                self._phase_callables[phase_obj.name](
                    *(list(self.statevars.values()) + list(point))
                )

        # Add points and calculated energies to the DataFrame
        data_dict = {'GM':energies, 'Phase':phase_obj.name}
        data_dict.update(self.statevars)

        for comp in sorted(list(self.components)):
            if comp == 'VA':
                continue
            data_dict['X('+comp+')'] = [0 for n in range(len(points))]

        for column_idx, data in enumerate(points.T):
            data_dict[self._varnames[column_idx]] = data

        # Now map the internal degrees of freedom to global coordinates
        for p_idx, pts in enumerate(points):
            for idx, coordinate in enumerate(pts):
                cur_var = self._variables[idx]
                if cur_var.species == 'VA':
                    continue
                ratio = site_ratios[cur_var.sublattice_index]
                data_dict['X('+cur_var.species+')'][p_idx] += ratio*coordinate

        return data_dict

    def _build_objective_functions(self, dbf, ast):
        "Construct objective function callables for each phase."
        for phase_name, phase_obj in self.phases.items():
            # Build the symbolic representation of the energy
            mod = Model(dbf, list(self.components), phase_name)
            # Construct an ordered list of the variables
            self._variables = []
            self._sublattice_dof = []
            for idx, sublattice in enumerate(phase_obj.constituents):
                dof = 0
                for component in set(sublattice).intersection(self.components):
                    self._variables.append(
                        v.SiteFraction(phase_name, idx, component)
                        )
                    dof += 1
                self._sublattice_dof.append(dof)

            # Build the "fast" representation of that model
            self._phase_callables[phase_name] = make_callable(mod, \
                list(self.statevars.keys()) + self._variables, mode=ast)
            # Make user-friendly site fraction column labels
            self._varnames = ['Y('+variable.phase_name+',' + \
                    str(variable.sublattice_index) + ',' + \
                    variable.species +')' for variable in self._variables]

