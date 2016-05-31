from pycalphad import Database, Model, calculate, equilibrium
import pycalphad.variables as v
import sympy

class SoluteTrapModel(Model):
    def __init__(self, dbe, comps, phase,
                 a=None, b=None, n=1, x0=None, **kwargs):
        # Save extra parameters
        self.a = a
        self.b = b
        self.n = n
        self.x0 = x0
        # Initialize parent Model
        super(SoluteTrapModel, self).__init__(dbe, comps, phase, **kwargs)
    def build_phase(self, dbe, phase_name, symbols, param_search):
        phase = dbe.phases[phase_name]
        self.models['ref'] = self.reference_energy(phase, param_search)
        self.models['idmix'] = self.ideal_mixing_energy(phase, param_search)
        self.models['xsmix'] = self.excess_mixing_energy(phase, param_search)
        self.models['mag'] = self.magnetic_energy(phase, param_search)

        # Here is where we add our custom contribution
        self.models['soltrap'] = self.solute_trapping_energy(phase)

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
    def solute_trapping_energy(self, phase):
        if self.x0 is None:
            raise ValueError('x0 must be specified for SoluteTrapModel')
        else:
            x0 = self.x0
        a = self.a if self.a is not None else sympy.S.Zero
        b = self.b if self.b is not None else sympy.S.Zero
        n = self.n if self.n is not None else sympy.S.One
        a = sympy.S(a)
        b = sympy.S(b)
        n = sympy.S(n)
        result = sympy.S.Zero
        # Augmented assignment prevents mutating self.a
        result += a
        second_term = sympy.S.Zero
        for component, value in x0.items():
            second_term += sympy.Abs(self.standard_mole_fraction(component) - value) ** n
        if n is not sympy.S.One:
            second_term **= (1./n)
        second_term *= b
        result += second_term
        #print('Extra term', result / self._site_ratio_normalization(phase))
        return result / self._site_ratio_normalization(phase)

def run_test():
    dbf = Database('Fe-C_Fei_Brosh_2014_09.TDB')
    comps = ['FE', 'C', 'VA']
    phases = ['FCC_A1', 'LIQUID']
    conds = {v.T: 500, v.P: 101325, v.X('C'): 0.1}
    x0 = {'FE': 0.7, 'C': 0.3}
    a = 0
    b = -10 * v.T
    slmod = SoluteTrapModel(dbf, comps, 'FCC_A1',
                            a=a,  b=b, n=1, x0=x0)
    # Use custom model for fcc; use default for all others
    models = {'FCC_A1': slmod}
    eq = equilibrium(dbf, comps, phases, conds, model=models)
    print(eq)
    res = calculate(dbf, comps, 'FCC_A1', T=conds[v.T], P=conds[v.P],
                    model=models)
    res_nosoltrap = calculate(dbf, comps, 'FCC_A1', T=conds[v.T], P=conds[v.P])
    import matplotlib.pyplot as plt
    plt.scatter(res.X.sel(component='C'), res.GM, c='r')
    plt.scatter(res.X.sel(component='C'), res_nosoltrap.GM, c='k')
    plt.xlabel('Mole Fraction C')
    plt.ylabel('Molar Gibbs Energy')
    plt.title('T = {} K'.format(conds[v.T]))
    plt.savefig('fcc_energy.png')

if __name__ == "__main__":
    run_test()