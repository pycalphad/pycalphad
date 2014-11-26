"""
The conditions module defines convenience classes for specifying
equilibrium conditions.
"""

import pycalphad.variables as v

class Condition(object):
    "A generic condition of equilibrium."
    def __init__(self, equation):
        """
        Construct a generic equilibrium condition.

        Parameters
        ----------
        equation : SymPy object
            Independent condition in the form: `equation` = 0
        """
        self.equation = equation
        self.variables = equation.atoms(v.StateVariable)
    def __repr__(self):
        return 'Condition(%s)' % self.equation

class FixedVariable(Condition):
    "Fixed variable, e.g., total number of moles, temperature, composition."
    def __init__(self, symbol, value):
        Condition.__init__(self, symbol - value)
        self.symbol = symbol
        self.value = value
    def __repr__(self):
        return 'FixedVariable(%s, %.8g)' % (self.symbol, self.value)
    def __str__(self):
        return '%s = %.4g (fixed)' % (self.symbol, self.value)

class FixedPartialMolarQuantity(Condition):
    "Fixed partial molar quantity, e.g., chemical potential of a component."
    def __init__(self):
        Condition.__init__(self, None, None)
        raise NotImplementedError
