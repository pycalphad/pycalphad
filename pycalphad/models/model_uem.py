"""
UEM (Unified Extrapolation Model) Thermodynamic Model Implementation

This module provides a pycalphad-compatible implementation of the Unified Extrapolation
Model (UEM) for calculating thermodynamic properties of multicomponent solution phases.

The UEM model extrapolates from binary subsystem data to predict multicomponent behavior
by introducing effective mole fractions that account for component similarity.

Key Features:
- Compatible with standard pycalphad workflow
- Uses only binary interaction parameters (no ternary+ parameters needed)
- Provides smooth extrapolation across composition space
- Reduces to standard models for binary systems

References:
- Chou, K. C. (2020). On the definition of the components' difference in properties
  in the unified extrapolation model. Fluid Phase Equilibria.
- Chou, K. C. (2024). Latest UEM formulations. Thermochimica Acta.
"""
from pycalphad import Model
from pycalphad import variables as v
from pycalphad.core.utils import wrap_symbol
from sympy import S, Float, exp, Add
from tinydb import where
import logging
import pycalphad.models.uem_symbolic as uem

# Configure logging
logger = logging.getLogger(__name__)


class ModelUEM(Model):
    """
    PyCalphad-compatible Unified Extrapolation Model (UEM) for solution phases.

    The UEM calculates excess Gibbs energy of multicomponent solution phases by
    extrapolating from binary subsystem data using effective mole fractions that
    account for component similarities.

    The model follows the standard pycalphad Model architecture with contributions:
    - Reference energy: Pure component endmember energies
    - Ideal mixing energy: Configurational entropy of mixing
    - Excess mixing energy: UEM-based excess Gibbs energy from binary parameters

    Parameters
    ----------
    dbe : Database
        Thermodynamic database containing phase and parameter information
    comps : list of str
        Component names to consider in the model
    phase_name : str
        Name of the phase to model
    parameters : dict or list, optional
        Optional dictionary of parameters to substitute in the model

    Attributes
    ----------
    components : set
        Set of active components
    constituents : list
        List of constituent sets for each sublattice
    phase_name : str
        Name of the phase being modeled

    Examples
    --------
    >>> from pycalphad import Database, calculate, variables as v
    >>> from pycalphad.models.model_uem import ModelUEM
    >>> dbf = Database('multicomponent.tdb')
    >>> comps = ['AL', 'CR', 'NI', 'VA']
    >>> phases = ['LIQUID']
    >>> # Calculate using UEM model
    >>> result = calculate(dbf, comps, phases, model=ModelUEM,
    ...                   T=1800, P=101325, N=1)

    Notes
    -----
    - The UEM requires binary interaction parameters for all component pairs
    - Missing binary parameters are treated as ideal (zero excess)
    - For binary systems, UEM gives identical results to standard Redlich-Kister
    - The model automatically handles vacancies (VA) by excluding them from excess energy
    - Numerical stability is maintained through careful handling of edge cases

    See Also
    --------
    pycalphad.Model : Base model class
    pycalphad.models.uem_symbolic : UEM symbolic expression builders
    """

    # Define energy contribution terms
    # This follows the standard pycalphad Model architecture
    contributions = [
        ('ref', 'reference_energy'),      # Pure component energies
        ('idmix', 'ideal_mixing_energy'), # Ideal entropy of mixing
        ('xsmix', 'excess_mixing_energy') # UEM excess energy (overridden)
    ]

    def excess_mixing_energy(self, dbe):
        """
        Calculate excess mixing energy using the UEM formulation.

        This method overrides the standard excess mixing energy calculation to use
        the Unified Extrapolation Model instead of the traditional Redlich-Kister-
        Muggianu approach.

        Parameters
        ----------
        dbe : Database
            Thermodynamic database containing binary interaction parameters

        Returns
        -------
        SymPy expression
            Excess Gibbs energy expression (J/mol of formula unit)

        Notes
        -----
        - Vacancies (VA) are automatically excluded from the calculation
        - The expression is normalized by site ratio normalization factor
        - Uses binary parameters only; ternary+ parameters are ignored
        """
        # Get list of components excluding vacancies
        comps = [str(c) for c in self.components if str(c) != 'VA']

        if len(comps) < 2:
            # Single component: no excess mixing
            logger.debug(f"Single component system in {self.phase_name}, no excess mixing")
            return S.Zero

        # Build UEM excess Gibbs energy expression
        logger.info(f"Building UEM excess energy for {len(comps)}-component system: {comps}")
        expr = uem.get_uem1_excess_gibbs_expr(dbe, comps, self.phase_name, v.T)

        # Normalize by site ratios (required for sublattice models)
        expr = expr / self._site_ratio_normalization

        return expr

    def reference_energy(self, dbe):
        """
        Calculate reference energy contribution.

        Inherited from parent Model class. Returns the weighted sum of pure
        component endmember energies.

        Parameters
        ----------
        dbe : Database
            Thermodynamic database

        Returns
        -------
        SymPy expression
            Reference energy expression (J/mol of formula unit)
        """
        return super().reference_energy(dbe)

    def ideal_mixing_energy(self, dbe):
        """
        Calculate ideal mixing energy contribution.

        Inherited from parent Model class. Returns the configurational entropy
        of mixing: -T*S_config where S_config = -R*sum(y_i * ln(y_i)).

        Parameters
        ----------
        dbe : Database
            Thermodynamic database

        Returns
        -------
        SymPy expression
            Ideal mixing energy expression (J/mol of formula unit)
        """
        return super().ideal_mixing_energy(dbe)


class DummyModel(Model):
    """
    简单的测试模型，用于验证框架
    """
    contributions = [
        ('ref', 'reference_energy'),
        ('idmix', 'ideal_mixing_energy'),
        ('xsmix', 'excess_mixing_energy')
    ]

    def excess_mixing_energy(self, dbe):
        """简单的过剩能测试函数"""
        logger.info("使用DummyModel")
        comps = [str(c) for c in self.components if str(c) != 'VA']
        
        if len(comps) < 2:
            return Float(0.0)
        
        # 简单的二元相互作用
        if 'AL' in comps and 'NI' in comps:
            return v.X('AL') * v.X('NI') * Float(-10000.0)
        
        return Float(0.0)