import numpy as np
import copy

from pycalphad import Workspace, calculate, variables as v
from pycalphad.core.composition_set import CompositionSet
from pycalphad.core.solver import Solver
from pycalphad.property_framework.metaproperties import DormantPhase

from mapping.primitives import STATEVARS

def _update_cs_phase_frac(comp_set, phase_frac):
    comp_set.update(comp_set.dof[len(STATEVARS):], phase_frac, comp_set.dof[:len(STATEVARS)])

def _get_statevars_array(conditions):
    return np.asarray([conditions[sv] for sv in STATEVARS], dtype=np.float_)

def _fix_composition_set(comp_sets, comp_sets_to_fix):
    #Normalization for phase fractions of unfixed phases
    #Fixed phases will be fixed at a phase fraction of 0
    #So we adjust so that sum(phase fractions) = 1
    tot_phase_frac = 0
    for cs in comp_sets:
        if cs in comp_sets_to_fix:
            cs.fixed = True
            _update_cs_phase_frac(cs, 0)
        else:
            tot_phase_frac += cs.NP

    for cs in comp_sets:
        if not cs.fixed:
            _update_cs_phase_frac(cs, cs.NP/tot_phase_frac)

class ZPFStepper:
    def __init__(self, dbf, comps, phases, conditions, models = None, phase_records = None):
        self.dbf = dbf
        self.components = comps
        self.phases = phases
        self.conditions = conditions
        self.models = models
        self.phase_records = phase_records

        #Create composition sets, these will be used for stepping
        wks = Workspace(dbf, comps, phases, conditions)
        if self.models is None:
            self.models = wks.models.unwrap()
        if self.phase_records is None:
            self.phase_records = wks.phase_record_factory

        self.chemical_potentials = np.squeeze(wks.eq.MU)
        for _, cs in wks.enumerate_composition_sets():
            self.comp_sets = cs

        self.conditions = {key: key.compute_property(self.comp_sets, self.conditions, self.chemical_potentials) for key in self.conditions.keys()}

        #Store number of phases as a quick way to check if a phase became unstable
        self.num_phases = len(self.comp_sets)

    def fix_composition_set(self):
        if len(self.comp_sets) > 1:
            _fix_composition_set(self.comp_sets, [self.comp_sets[0]])

    def get_conditions_copy(self):
        return copy.deepcopy(self.conditions)
    
    def get_composition_sets_copy(self):
        return [cs for cs in self.comp_sets]

    def update_equilibrium_with_new_conditions(self, new_conditions, free_var = None):
        '''
        Pretty much a copy of the old update with new conditions

        Returns
        -------
        True - number of phases stayed constant
        False - number of phases reduced
        None - equilibrium failed
        '''
        new_state_conds = _get_statevars_array(new_conditions)
        for cs in self.comp_sets:
            cs.update(cs.dof[len(STATEVARS):], cs.NP, new_state_conds)

        if free_var is not None:
            del new_conditions[free_var]

        # Main change from previous mapper is that we don't create a copy of the composition sets
        # 1. Phases becoming unstable are tracked by check against the original number of phases
        #    The ZPF stepper is only responsible for a single ZPF line, so once the number of phases
        #    change, we create a new ZPF stepper for the new ZPF line
        # 2. We will record the each state of the ZPF line so that we can always go back if needed
        solver = Solver(remove_metastable=True, allow_changing_phases=False)
        result = solver.solve(self.comp_sets, new_conditions)
        
        if any(np.isnan(result.chemical_potentials)):
            return None

        if free_var is not None:
            new_conditions[free_var] = free_var.compute_property(self.comp_sets, new_conditions, result.chemical_potentials)

        self.conditions = copy.deepcopy(new_conditions)
        self.chemical_potentials = np.array(result.chemical_potentials)

        return len(self.comp_sets) == self.num_phases

    def check_if_global_min(self, tol = 1e-5):
        '''
        For each possible phase:
            1. Sample DOF and find CS that minimizes driving force
            2. Create a DormantPhase with CS and compute driving force with potentials at equilibrium
            3. If driving force is negative, then new phase is stable
            4. Check that the new CS doesn't match with a currently stable CS
            4. Hope that this works on miscibility gaps

        This should take care of the DLASCLS error since we compute the new phase separately so if
        the composition clashes with a fixed phase, we check that afterwards before attempting to
        run equilibrium on two CS with the same composition
        '''
        pdens = 500
        #Get driving force and find index that maximized driving force
        state_conds = {str(key): self.conditions[key] for key in STATEVARS}
        points = calculate(self.dbf, self.components, self.phases, model=self.models, phase_records=self.phase_records, output='GM', to_xarray=False, pdens=pdens, **state_conds)
        gm = np.squeeze(points.GM)
        x = np.squeeze(points.X)
        y = np.squeeze(points.Y)
        phase_ids = np.squeeze(points.Phase)
        g_chempot = x * self.chemical_potentials
        dG = np.sum(g_chempot, axis=1) - gm

        max_id = np.argmax(dG)

        #Create composition set and create DormantPhase
        cs = CompositionSet(self.phase_records[phase_ids[max_id]])
        cs.update(y[max_id, :cs.phase_record.phase_dof], 1.0, _get_statevars_array(self.conditions))
        dormantPhase = DormantPhase(cs, None)
        dG = dormantPhase.driving_force.compute_property(self.comp_sets, self.conditions, self.chemical_potentials)
        
        if dG < tol:
            return None
        else:
            return cs