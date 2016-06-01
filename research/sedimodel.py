from pycalphad import Database, Model, calculate, equilibrium
import numpy as np
import pycalphad.variables as v
import sympy
from tinydb import where

class EinsteinModel(Model):
    def build_phase(self, dbe, phase_name, symbols, param_search):
        phase = dbe.phases[phase_name]
        self.models['ref'] = self.reference_energy(phase, param_search)
        self.models['idmix'] = self.ideal_mixing_energy(phase, param_search)
        self.models['xsmix'] = self.excess_mixing_energy(phase, param_search)
        self.models['mag'] = self.magnetic_energy(phase, param_search)

        # Here is where we add our custom contribution
        self.models['einstein'] = self.einstein_energy(phase, param_search)

        # Extra code necessary for compatibility with order-disorder model
        ordered_phase_name = None
        disordered_phase_name = None
        try:
            ordered_phase_name = phase.model_hints['ordered_phase']
            disordered_phase_name = phase.model_hints['disordered_phase']
        except KeyError:
            pass
        if ordered_phase_name == phase_name:
            self.models['ord'] = self.atomic_ordering_energy(dbe,
                                                             disordered_phase_name,
                                                             ordered_phase_name)
    def einstein_energy(self, phase, param_search):
        einstein_param_query = (
            (where('phase_name') == phase.name) & \
            (where('parameter_type') == 'THETA') & \
            (where('constituent_array').test(self._array_validity))
        )
        theta = self.redlich_kister_sum(phase, param_search, einstein_param_query)
        self.TE = self.einstein_temperature = theta
        x = theta / v.T
        self.testprop = 3.0 * v.R * ((x**2) * sympy.exp(x) / ((sympy.exp(x) - 1.0) ** 2))
        result = 3 * v.R * theta * (0.5 + 1./(sympy.exp(x) - 1))
        return result

def run_test():
    dbf = Database()
    dbf.elements = frozenset(['A'])
    dbf.add_phase('TEST', {}, [1])
    dbf.add_phase_constituents('TEST', [['A']])
    # add THETA parameters here
    dbf.add_parameter('THETA', 'TEST', [['A']], 0, 334.)
    conds = {v.T: np.arange(1.,800.,1), v.P: 101325}
    res = calculate(dbf, ['A'], 'TEST', T=conds[v.T], P=conds[v.P],
                    model=EinsteinModel, output='testprop')
    #res_TE = calculate(dbf, ['A'], 'TEST', T=conds[v.T], P=conds[v.P],
    #                model=EinsteinModel, output='einstein_temperature')
    import matplotlib.pyplot as plt
    plt.scatter(res['T'], res['testprop'])
    plt.xlabel('Temperature (K)')
    plt.ylabel('Molar Heat Capacity (J/mol-K)')
    plt.savefig('einstein.png')

if __name__ == "__main__":
    run_test()