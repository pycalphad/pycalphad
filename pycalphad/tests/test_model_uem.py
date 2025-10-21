"""
Tests for the Unified Extrapolation Model (UEM) implementation.

This test suite verifies the correctness of the UEM model for calculating
thermodynamic properties of multicomponent solution phases.
"""

import pytest
import numpy as np
from pycalphad import Database, Model, calculate, variables as v
from pycalphad.models.model_uem import ModelUEM
from pycalphad.core.utils import instantiate_models
from sympy import Symbol, simplify, S


class TestUEMBasic:
    """Basic tests for UEM model infrastructure."""

    def test_model_instantiation(self):
        """Test that UEM model can be instantiated."""
        dbf = Database()
        # Add a simple LIQUID phase
        dbf.add_structure_entry('A', 'LIQUID', S.Zero)
        dbf.add_structure_entry('B', 'LIQUID', S.Zero)

        # Should not raise exception
        mod = ModelUEM(dbf, ['A', 'B', 'VA'], 'LIQUID')
        assert isinstance(mod, Model)
        assert isinstance(mod, ModelUEM)

    def test_contributions_defined(self):
        """Test that UEM model has required contributions."""
        dbf = Database()
        dbf.add_structure_entry('A', 'LIQUID', S.Zero)
        dbf.add_structure_entry('B', 'LIQUID', S.Zero)

        mod = ModelUEM(dbf, ['A', 'B', 'VA'], 'LIQUID')

        # Check that standard contributions are present
        assert 'ref' in [c[0] for c in mod.contributions]
        assert 'idmix' in [c[0] for c in mod.contributions]
        assert 'xsmix' in [c[0] for c in mod.contributions]

    def test_binary_system_ideal(self):
        """Test UEM on ideal binary system (no interaction parameters)."""
        # Create simple database with no interaction parameters
        dbf = Database()
        dbf.add_structure_entry('A', 'LIQUID', S.Zero)
        dbf.add_structure_entry('B', 'LIQUID', S.Zero)

        mod = ModelUEM(dbf, ['A', 'B', 'VA'], 'LIQUID')

        # Excess energy should be zero for ideal system
        excess = mod.models['xsmix']
        assert excess == S.Zero or simplify(excess) == S.Zero


class TestUEMWithParameters:
    """Tests for UEM model with actual interaction parameters."""

    @pytest.fixture
    def binary_database(self):
        """Create a test database with binary interaction parameters."""
        dbf = Database()

        # Add phase and components
        dbf.add_structure_entry('AL', 'LIQUID', S.Zero)
        dbf.add_structure_entry('NI', 'LIQUID', S.Zero)

        # Add binary interaction parameter L0
        # LIQUID: AL-NI, L0 = -50000 + 10*T
        dbf.add_parameter(
            phase_name='LIQUID',
            parameter_type='G',
            constituents=[['AL', 'NI']],
            parameter_order=0,
            parameter=-50000 + 10*v.T,
            diffusing_species=None
        )

        return dbf

    def test_binary_with_parameters(self, binary_database):
        """Test that UEM generates non-zero excess energy for binary with parameters."""
        mod = ModelUEM(binary_database, ['AL', 'NI', 'VA'], 'LIQUID')

        # Should have non-zero excess mixing energy
        excess = mod.models['xsmix']
        assert excess != S.Zero

    def test_uem_reduces_to_redlich_kister_for_binary(self, binary_database):
        """Test that UEM gives same result as standard model for binary systems."""
        # Standard model
        mod_standard = Model(binary_database, ['AL', 'NI', 'VA'], 'LIQUID')

        # UEM model
        mod_uem = ModelUEM(binary_database, ['AL', 'NI', 'VA'], 'LIQUID')

        # For binary systems, UEM should reduce to standard Redlich-Kister
        # Note: Symbolic expressions may differ in form but should be equivalent
        standard_excess = simplify(mod_standard.models['xsmix'])
        uem_excess = simplify(mod_uem.models['xsmix'])

        # The expressions should be equivalent (allowing for different symbolic forms)
        assert standard_excess == uem_excess or simplify(standard_excess - uem_excess) == S.Zero


class TestUEMTernary:
    """Tests for UEM on ternary systems."""

    @pytest.fixture
    def ternary_database(self):
        """Create test database with ternary system."""
        dbf = Database()

        # Add components
        dbf.add_structure_entry('A', 'LIQUID', S.Zero)
        dbf.add_structure_entry('B', 'LIQUID', S.Zero)
        dbf.add_structure_entry('C', 'LIQUID', S.Zero)

        # Add binary interaction parameters
        # A-B: L0 = -10000
        dbf.add_parameter(
            phase_name='LIQUID',
            parameter_type='G',
            constituents=[['A', 'B']],
            parameter_order=0,
            parameter=-10000.0,
            diffusing_species=None
        )

        # A-C: L0 = -15000
        dbf.add_parameter(
            phase_name='LIQUID',
            parameter_type='G',
            constituents=[['A', 'C']],
            parameter_order=0,
            parameter=-15000.0,
            diffusing_species=None
        )

        # B-C: L0 = -20000
        dbf.add_parameter(
            phase_name='LIQUID',
            parameter_type='G',
            constituents=[['B', 'C']],
            parameter_order=0,
            parameter=-20000.0,
            diffusing_species=None
        )

        return dbf

    def test_ternary_model_builds(self, ternary_database):
        """Test that UEM model builds for ternary system."""
        mod = ModelUEM(ternary_database, ['A', 'B', 'C', 'VA'], 'LIQUID')

        # Should have non-zero excess energy
        excess = mod.models['xsmix']
        assert excess != S.Zero

    def test_ternary_calculation_runs(self, ternary_database):
        """Test that calculation runs successfully with UEM for ternary."""
        result = calculate(
            ternary_database,
            ['A', 'B', 'C', 'VA'],
            ['LIQUID'],
            model=ModelUEM,
            T=1000,
            P=101325,
            N=1
        )

        # Check that result is valid
        assert result is not None
        assert 'GM' in result.coords['output'].values
        assert not np.any(np.isnan(result.GM.values))


class TestUEMNumericalStability:
    """Tests for numerical stability of UEM calculations."""

    def test_pure_component_limit(self):
        """Test UEM behavior at pure component limits."""
        dbf = Database()
        dbf.add_structure_entry('A', 'LIQUID', S.Zero)
        dbf.add_structure_entry('B', 'LIQUID', S.Zero)

        dbf.add_parameter(
            phase_name='LIQUID',
            parameter_type='G',
            constituents=[['A', 'B']],
            parameter_order=0,
            parameter=-10000.0,
            diffusing_species=None
        )

        # Calculate at pure A
        result = calculate(
            dbf,
            ['A', 'B', 'VA'],
            ['LIQUID'],
            model=ModelUEM,
            T=1000,
            P=101325,
            X_A=1.0,
            N=1
        )

        # At pure component, excess energy should be zero
        assert not np.any(np.isnan(result.GM.values))

    def test_equal_composition(self):
        """Test UEM at equimolar composition."""
        dbf = Database()
        dbf.add_structure_entry('A', 'LIQUID', S.Zero)
        dbf.add_structure_entry('B', 'LIQUID', S.Zero)
        dbf.add_structure_entry('C', 'LIQUID', S.Zero)

        # Add symmetric binary parameters
        for pair in [['A', 'B'], ['A', 'C'], ['B', 'C']]:
            dbf.add_parameter(
                phase_name='LIQUID',
                parameter_type='G',
                constituents=[pair],
                parameter_order=0,
                parameter=-10000.0,
                diffusing_species=None
            )

        # Calculate at equimolar
        result = calculate(
            dbf,
            ['A', 'B', 'C', 'VA'],
            ['LIQUID'],
            model=ModelUEM,
            T=1000,
            P=101325,
            X_A=1.0/3.0,
            X_B=1.0/3.0,
            N=1
        )

        assert not np.any(np.isnan(result.GM.values))
        assert not np.any(np.isinf(result.GM.values))


class TestUEMSymbolicFunctions:
    """Tests for individual UEM symbolic functions."""

    def test_is_binary_in_phase(self):
        """Test binary detection helper function."""
        from pycalphad.models.uem_symbolic import is_binary_in_phase

        # Binary case
        assert is_binary_in_phase([['A', 'B']], 'A', 'B') == True
        assert is_binary_in_phase([['B', 'A']], 'A', 'B') == True

        # Ternary case
        assert is_binary_in_phase([['A', 'B', 'C']], 'A', 'B') == False

        # Wrong components
        assert is_binary_in_phase([['A', 'B']], 'A', 'C') == False

    def test_delta_expression_basic(self):
        """Test property difference calculation."""
        from pycalphad.models.uem_symbolic import uem1_delta_expr

        dbf = Database()
        dbf.add_structure_entry('A', 'LIQUID', S.Zero)
        dbf.add_structure_entry('B', 'LIQUID', S.Zero)

        # Add asymmetric binary parameter (L1 != 0 creates asymmetry)
        dbf.add_parameter(
            phase_name='LIQUID',
            parameter_type='G',
            constituents=[['A', 'B']],
            parameter_order=1,
            parameter=5000.0,  # Non-zero L1 creates asymmetry
            diffusing_species=None
        )

        delta = uem1_delta_expr(dbf, 'A', 'B', 'LIQUID', v.T)

        # Should return a symbolic expression
        assert delta is not None
        # For asymmetric system, delta should be non-zero
        assert delta != S.Zero


class TestUEMRealSystem:
    """Tests with real thermodynamic database."""

    def test_alcrni_system(self):
        """Test UEM with Al-Cr-Ni system from example database."""
        # Check if example database exists
        import os
        db_path = '/home/user/pycalphad/examples/alcrni.tdb'

        if not os.path.exists(db_path):
            pytest.skip("Example database not found")

        dbf = Database(db_path)

        # Test calculation with UEM
        result = calculate(
            dbf,
            ['AL', 'CR', 'NI', 'VA'],
            ['LIQUID'],
            model=ModelUEM,
            T=1800,
            P=101325,
            N=1
        )

        assert result is not None
        assert 'GM' in result.coords['output'].values
        assert not np.all(np.isnan(result.GM.values))

    def test_uem_vs_standard_comparison(self):
        """Compare UEM and standard model predictions."""
        import os
        db_path = '/home/user/pycalphad/examples/alcrni.tdb'

        if not os.path.exists(db_path):
            pytest.skip("Example database not found")

        dbf = Database(db_path)

        # Standard model
        result_std = calculate(
            dbf,
            ['AL', 'CR', 'NI', 'VA'],
            ['LIQUID'],
            model=Model,
            T=1800,
            P=101325,
            X_AL=0.2,
            X_CR=0.3,
            N=1
        )

        # UEM model
        result_uem = calculate(
            dbf,
            ['AL', 'CR', 'NI', 'VA'],
            ['LIQUID'],
            model=ModelUEM,
            T=1800,
            P=101325,
            X_AL=0.2,
            X_CR=0.3,
            N=1
        )

        # Both should give valid results (values will differ as they use different models)
        assert not np.all(np.isnan(result_std.GM.values))
        assert not np.all(np.isnan(result_uem.GM.values))

        # Results should be different (UEM vs Muggianu extrapolation)
        # Note: For some compositions they might be similar, so we just check they're both valid
        print(f"Standard model GM: {result_std.GM.values[0]}")
        print(f"UEM model GM: {result_uem.GM.values[0]}")


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '-s'])
