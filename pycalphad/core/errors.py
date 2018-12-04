class CalculateError(Exception):
    "Exception related to use of calculate() function."


class DofError(Exception):
    "Error due to missing degrees of freedom."
    pass


class EquilibriumError(Exception):
    "Exception related to calculation of equilibrium."
    pass


class ConditionError(CalculateError, EquilibriumError):
    "Exception related to calculation conditions."
    pass
