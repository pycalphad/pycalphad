"""
Redlich-Kister-UEM Thermodynamic Model Implementation

This module provides a pycalphad-compatible implementation of the Redlich-Kister-UEM
method for calculating thermodynamic properties of multicomponent solution phases.

Method Overview:
---------------
In CALPHAD modeling, two steps are needed to describe multicomponent solutions:

Step 1: Binary Interaction Description
   - Uses Redlich-Kister polynomials: G_ex^ij = x_i*x_j * Σ_n L^n_ij*(x_i-x_j)^n
   - Parameters L^n_ij are obtained from binary phase diagram assessments
   - This step is IDENTICAL for all methods (Muggianu, Kohler, Toop, UEM)

Step 2: Multicomponent Extrapolation (from binary to ternary, quaternary, etc.)
   - Traditional: Redlich-Kister-Muggianu (symmetric geometric averaging)
   - Traditional: Redlich-Kister-Kohler (asymmetric geometric averaging)
   - Traditional: Redlich-Kister-Toop (asymmetric with designated component)
   - This module: Redlich-Kister-UEM (property-difference-based extrapolation)

UEM Extrapolation Principle:
---------------------------
Instead of geometric averaging, UEM calculates "effective mole fractions" for each
binary pair based on how similar the third (and higher) components are to the pair.
Similarity is quantified by property differences (δ) derived from binary parameters.

Key Advantages over Muggianu:
- Physical basis (property differences) vs arbitrary geometry
- Better for highly asymmetric systems
- Includes Muggianu, Kohler, Toop as special limiting cases
- Uses ONLY binary parameters (no ternary terms needed)

Correct Terminology:
-------------------
- NOT: "UEM model" vs "Redlich-Kister model"
- CORRECT: "Redlich-Kister-UEM" vs "Redlich-Kister-Muggianu"
- Binary description: Same Redlich-Kister polynomials for both
- Difference: Only in how binary data is extrapolated to multicomponent

Key Features:
------------
- Compatible with standard pycalphad workflow
- Uses same binary Redlich-Kister parameters as traditional models
- Provides alternative extrapolation to multicomponent systems
- Smooth predictions across composition space
- Reduces to standard Redlich-Kister for binary systems

References:
----------
- Chou, K. C. (2020). Fluid Phase Equilibria, 507, 112416.
- Chou, K. C., Wei, S. K. (2020). J. Molecular Liquids, 298, 111951.
- Chou, K. C. et al. (2024). Thermochimica Acta, 179824.
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
    PyCalphad-compatible Redlich-Kister-UEM model for solution phases.

    This model implements the UEM extrapolation method for multicomponent systems.
    It uses the SAME Redlich-Kister polynomials for binary interactions as the
    standard Model class, but employs a DIFFERENT extrapolation scheme to predict
    multicomponent behavior.

    Comparison with Standard Model:
    ------------------------------
    Standard Model (pycalphad.Model):
        Binary: Redlich-Kister polynomials
        Extrapolation: Muggianu (symmetric geometric averaging)

    This Model (ModelUEM):
        Binary: Redlich-Kister polynomials (SAME as standard)
        Extrapolation: UEM (property-difference-based)

    Energy Contributions:
    --------------------
    - Reference energy: Pure component endmember energies (same as standard)
    - Ideal mixing energy: Configurational entropy -T*S_config (same as standard)
    - Excess mixing energy: Redlich-Kister-UEM extrapolation (DIFFERENT from standard)

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
    - Uses standard Redlich-Kister polynomials for binary interactions (L0, L1, L2, ...)
    - Only differs from standard Model in multicomponent extrapolation method
    - For binary systems: Identical to standard Model (pure Redlich-Kister)
    - For ternary+: UEM extrapolation instead of Muggianu extrapolation
    - Missing binary parameters are treated as ideal (zero excess)
    - Vacancies (VA) are automatically excluded from excess energy calculations
    - Numerical stability ensured through careful handling of edge cases

    Terminology:
    -----------
    - Correct: "Redlich-Kister-UEM" vs "Redlich-Kister-Muggianu"
    - NOT: "UEM" vs "Redlich-Kister"
    - Both use Redlich-Kister for binaries; differ only in multicomponent extrapolation

    See Also
    --------
    pycalphad.Model : Standard model (Redlich-Kister-Muggianu)
    pycalphad.models.uem_symbolic : UEM extrapolation symbolic functions
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
        Calculate excess mixing energy using Redlich-Kister-UEM extrapolation.

        This method overrides the standard excess mixing energy calculation to use
        UEM extrapolation instead of Muggianu extrapolation for multicomponent systems.

        Important: Binary Interactions
        ------------------------------
        - Uses the SAME Redlich-Kister polynomial form as standard Model
        - Queries the SAME binary parameters (L0, L1, L2, ...) from database
        - For binary systems: Results are IDENTICAL to standard Model

        Difference: Multicomponent Extrapolation
        ----------------------------------------
        - Standard Model: Muggianu extrapolation (geometric symmetric averaging)
        - This Model: UEM extrapolation (property-difference-based effective mole fractions)

        Parameters
        ----------
        dbe : Database
            Thermodynamic database containing binary Redlich-Kister parameters

        Returns
        -------
        SymPy expression
            Excess Gibbs energy expression (J/mol of formula unit)
            Using Redlich-Kister-UEM extrapolation

        Notes
        -----
        - Binary interactions: Standard Redlich-Kister polynomials
        - Multicomponent: UEM extrapolation method
        - Vacancies (VA) are automatically excluded
        - Expression normalized by site ratio normalization factor
        - Uses only binary parameters; ternary+ parameters ignored
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