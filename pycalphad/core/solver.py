import numpy as np
from collections import namedtuple
from pycalphad.core.minimizer import SystemSpecification

SolverResult = namedtuple('SolverResult', ['converged', 'x', 'chemical_potentials'])

class SolverBase(object):
    """"Base class for solvers."""
    ignore_convergence = False
    def solve(self, composition_sets, conditions):
        """
        *Implement this method.*
        Minimize the energy under the specified conditions using the given candidate composition sets.

        Parameters
        ----------
        composition_sets : List[pycalphad.core.composition_set.CompositionSet]
            List of CompositionSet objects in the starting point. Modified in place.
        conditions : OrderedDict[str, float]
            Conditions to satisfy.

        Returns
        -------
        pycalphad.core.solver.SolverResult
        """
        raise NotImplementedError("A subclass of Solver must be implemented.")


class Solver(SolverBase):
    def __init__(self, verbose=False, remove_metastable=True, **options):
        self.verbose = verbose
        self.remove_metastable = remove_metastable


    def get_system_spec(self, composition_sets, conditions):
        """
        Create a SystemSpecification object for the specified conditions.

        Parameters
        ----------
        composition_sets : List[pycalphad.core.composition_set.CompositionSet]
            List of CompositionSet objects in the starting point. Modified in place.
        conditions : OrderedDict[str, float]
            Conditions to satisfy.

        Returns
        -------
        SystemSpecification

        """
        compsets = composition_sets
        state_variables = compsets[0].phase_record.state_variables
        nonvacant_elements = compsets[0].phase_record.nonvacant_elements
        num_statevars = len(state_variables)
        num_components = len(nonvacant_elements)
        chemical_potentials = np.zeros(num_components)
        prescribed_elemental_amounts = []
        prescribed_element_indices = []
        for cond, value in conditions.items():
            if str(cond).startswith('X_'):
                el = str(cond)[2:]
                el_idx = list(nonvacant_elements).index(el)
                prescribed_elemental_amounts.append(float(value))
                prescribed_element_indices.append(el_idx)
        prescribed_element_indices = np.array(prescribed_element_indices, dtype=np.int32)
        prescribed_elemental_amounts = np.array(prescribed_elemental_amounts)
        prescribed_system_amount = conditions.get('N', 1.0)
        fixed_chemical_potential_indices = np.array([nonvacant_elements.index(key[3:]) for key in conditions.keys() if key.startswith('MU_')], dtype=np.int32)
        free_chemical_potential_indices = np.array(sorted(set(range(num_components)) - set(fixed_chemical_potential_indices)), dtype=np.int32)
        for fixed_chempot_index in fixed_chemical_potential_indices:
            el = nonvacant_elements[fixed_chempot_index]
            chemical_potentials[fixed_chempot_index] = conditions.get('MU_' + str(el))
        fixed_statevar_indices = []
        for statevar_idx, statevar in enumerate(state_variables):
            if str(statevar) in [str(k) for k in conditions.keys()]:
                fixed_statevar_indices.append(statevar_idx)
        free_statevar_indices = np.array(sorted(set(range(num_statevars)) - set(fixed_statevar_indices)), dtype=np.int32)
        fixed_statevar_indices = np.array(fixed_statevar_indices, dtype=np.int32)
        fixed_stable_compset_indices = np.array([i for i, compset in enumerate(compsets) if compset.fixed], dtype=np.int32)
        spec = SystemSpecification(num_statevars, num_components, prescribed_system_amount,
                                   chemical_potentials, prescribed_elemental_amounts,
                                   prescribed_element_indices,
                                   free_chemical_potential_indices, free_statevar_indices,
                                   fixed_chemical_potential_indices, fixed_statevar_indices,
                                   fixed_stable_compset_indices)
        return spec

    def solve(self, composition_sets, conditions):
        """
        Minimize the energy under the specified conditions using the given candidate composition sets.

        Parameters
        ----------
        composition_sets : List[pycalphad.core.composition_set.CompositionSet]
            List of CompositionSet objects in the starting point. Modified in place.
        conditions : OrderedDict[str, float]
            Conditions to satisfy.

        Returns
        -------
        SolverResult

        """
        spec = self.get_system_spec(composition_sets, conditions)
        state = spec.get_new_state(composition_sets)
        converged = spec.run_loop(state, 1000)

        if self.remove_metastable:
            phase_idx = 0
            compsets_to_remove = []
            for compset in composition_sets:
                # Mark unstable phases for removal
                if compset.NP <= 0.0 and not compset.fixed:
                    compsets_to_remove.append(int(phase_idx))
                phase_idx += 1
            # Watch removal order here, as the indices of composition_sets are changing!
            for idx in reversed(compsets_to_remove):
                del composition_sets[idx]

        phase_amt = [compset.NP for compset in composition_sets]

        x = composition_sets[0].dof
        state_variables = composition_sets[0].phase_record.state_variables
        num_statevars = len(state_variables)
        for compset in composition_sets[1:]:
            x = np.r_[x, compset.dof[num_statevars:]]
        x = np.r_[x, phase_amt]
        chemical_potentials = np.array(state.chemical_potentials)

        if self.verbose:
            print('Chemical Potentials', chemical_potentials)
            print(np.asarray(x))
        return SolverResult(converged=converged, x=x, chemical_potentials=chemical_potentials)
