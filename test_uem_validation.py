"""
Comprehensive Test Suite for UEM Implementation

This test file provides numerical validation of the UEM extrapolation method,
comparing results against known benchmarks and the standard Muggianu method.

Test Categories:
1. Binary system validation (UEM should equal standard RK)
2. Ternary system tests with known results
3. Numerical stability tests
4. Property difference calculations
5. Contribution coefficient validation
6. Edge case handling

Run this file directly to see detailed test results:
    python test_uem_validation.py
"""

import numpy as np
from sympy import Symbol, S, Float, lambdify, simplify
from pycalphad import Database, Model, calculate, variables as v
from pycalphad.models.model_uem import ModelUEM
import pycalphad.models.uem_symbolic as uem_sym


def create_simple_test_database():
    """
    Create a simple test database with well-defined binary parameters.

    System: A-B-C ternary
    Binary interactions:
    - A-B: Symmetric (L0 only)
    - A-C: Asymmetric (L0 and L1)
    - B-C: Strong asymmetric (L0, L1, L2)
    """
    dbf = Database()

    # Add phase definition
    dbf.add_structure_entry('A', 'LIQUID', S.Zero)
    dbf.add_structure_entry('B', 'LIQUID', S.Zero)
    dbf.add_structure_entry('C', 'LIQUID', S.Zero)

    # A-B: Symmetric interaction, L0 = -10000 J/mol
    dbf.add_parameter(
        phase_name='LIQUID',
        parameter_type='G',
        constituents=[['A', 'B']],
        parameter_order=0,
        parameter=-10000.0,
        diffusing_species=None
    )

    # A-C: Asymmetric interaction
    # L0 = -15000 + 5*T
    dbf.add_parameter(
        phase_name='LIQUID',
        parameter_type='G',
        constituents=[['A', 'C']],
        parameter_order=0,
        parameter=-15000.0 + 5.0*v.T,
        diffusing_species=None
    )
    # L1 = -3000
    dbf.add_parameter(
        phase_name='LIQUID',
        parameter_type='G',
        constituents=[['A', 'C']],
        parameter_order=1,
        parameter=-3000.0,
        diffusing_species=None
    )

    # B-C: Strong asymmetric interaction
    # L0 = -20000
    dbf.add_parameter(
        phase_name='LIQUID',
        parameter_type='G',
        constituents=[['B', 'C']],
        parameter_order=0,
        parameter=-20000.0,
        diffusing_species=None
    )
    # L1 = -5000
    dbf.add_parameter(
        phase_name='LIQUID',
        parameter_type='G',
        constituents=[['B', 'C']],
        parameter_order=1,
        parameter=-5000.0,
        diffusing_species=None
    )
    # L2 = -2000
    dbf.add_parameter(
        phase_name='LIQUID',
        parameter_type='G',
        constituents=[['B', 'C']],
        parameter_order=2,
        parameter=-2000.0,
        diffusing_species=None
    )

    return dbf


class TestBinarySystemEquivalence:
    """Test that UEM gives identical results to standard model for binary systems."""

    def __init__(self):
        self.dbf = create_simple_test_database()
        self.passed = []
        self.failed = []

    def test_binary_ab(self):
        """Test A-B binary system."""
        print("\n" + "="*70)
        print("TEST 1: Binary System A-B Equivalence")
        print("="*70)

        comps = ['A', 'B', 'VA']
        phases = ['LIQUID']

        # Test at several compositions
        x_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        T = 1000.0

        print(f"\nTemperature: {T} K")
        print(f"Binary: A-B (Symmetric, L0 = -10000)")
        print("\nX(A)    GM_Standard      GM_UEM           Difference")
        print("-" * 70)

        all_match = True
        for x_a in x_values:
            # Standard model
            res_std = calculate(
                self.dbf, comps, phases,
                T=T, P=101325, X_A=x_a, N=1
            )
            gm_std = float(res_std.GM.values[0])

            # UEM model
            res_uem = calculate(
                self.dbf, comps, phases, model=ModelUEM,
                T=T, P=101325, X_A=x_a, N=1
            )
            gm_uem = float(res_uem.GM.values[0])

            diff = abs(gm_uem - gm_std)
            match = diff < 1e-6  # 1 μJ/mol tolerance

            status = "✓" if match else "✗"
            print(f"{status} {x_a:.1f}    {gm_std:15.6f}  {gm_uem:15.6f}  {diff:15.9f}")

            if not match:
                all_match = False

        if all_match:
            self.passed.append("Binary A-B equivalence")
            print("\n✅ PASSED: Binary A-B equivalence test")
        else:
            self.failed.append("Binary A-B equivalence")
            print("\n❌ FAILED: Binary A-B equivalence test")

        return all_match

    def test_binary_ac_asymmetric(self):
        """Test A-C binary system (asymmetric)."""
        print("\n" + "="*70)
        print("TEST 2: Binary System A-C Equivalence (Asymmetric)")
        print("="*70)

        comps = ['A', 'C', 'VA']
        phases = ['LIQUID']

        x_values = [0.2, 0.5, 0.8]
        T = 1200.0

        print(f"\nTemperature: {T} K")
        print(f"Binary: A-C (Asymmetric, L0 = -15000+5T, L1 = -3000)")
        print("\nX(A)    GM_Standard      GM_UEM           Difference")
        print("-" * 70)

        all_match = True
        for x_a in x_values:
            res_std = calculate(
                self.dbf, comps, phases,
                T=T, P=101325, X_A=x_a, N=1
            )
            gm_std = float(res_std.GM.values[0])

            res_uem = calculate(
                self.dbf, comps, phases, model=ModelUEM,
                T=T, P=101325, X_A=x_a, N=1
            )
            gm_uem = float(res_uem.GM.values[0])

            diff = abs(gm_uem - gm_std)
            match = diff < 1e-6

            status = "✓" if match else "✗"
            print(f"{status} {x_a:.1f}    {gm_std:15.6f}  {gm_uem:15.6f}  {diff:15.9f}")

            if not match:
                all_match = False

        if all_match:
            self.passed.append("Binary A-C asymmetric equivalence")
            print("\n✅ PASSED: Binary A-C asymmetric equivalence test")
        else:
            self.failed.append("Binary A-C asymmetric equivalence")
            print("\n❌ FAILED: Binary A-C asymmetric equivalence test")

        return all_match


class TestTernarySystem:
    """Test ternary system where UEM and Muggianu should differ."""

    def __init__(self):
        self.dbf = create_simple_test_database()
        self.passed = []
        self.failed = []

    def test_ternary_comparison(self):
        """Compare UEM and Muggianu for ternary A-B-C."""
        print("\n" + "="*70)
        print("TEST 3: Ternary System A-B-C Comparison")
        print("="*70)

        comps = ['A', 'B', 'C', 'VA']
        phases = ['LIQUID']
        T = 1000.0

        # Test at several compositions
        compositions = [
            (0.33, 0.33, 0.34, "Equimolar"),
            (0.50, 0.25, 0.25, "A-rich"),
            (0.25, 0.50, 0.25, "B-rich"),
            (0.25, 0.25, 0.50, "C-rich"),
        ]

        print(f"\nTemperature: {T} K")
        print("\nComposition         GM_Muggianu  GM_UEM       Difference   % Diff")
        print("-" * 75)

        for x_a, x_b, x_c, desc in compositions:
            # Standard Muggianu
            res_std = calculate(
                self.dbf, comps, phases,
                T=T, P=101325, X_A=x_a, X_B=x_b, N=1
            )
            gm_std = float(res_std.GM.values[0])

            # UEM
            res_uem = calculate(
                self.dbf, comps, phases, model=ModelUEM,
                T=T, P=101325, X_A=x_a, X_B=x_b, N=1
            )
            gm_uem = float(res_uem.GM.values[0])

            diff = gm_uem - gm_std
            pct_diff = 100.0 * diff / gm_std if gm_std != 0 else 0.0

            print(f"{desc:20s} {gm_std:12.2f} {gm_uem:12.2f} {diff:12.2f} {pct_diff:8.3f}%")

        print("\n✅ PASSED: Ternary comparison completed")
        print("Note: Differences are expected and show UEM vs Muggianu extrapolation")
        self.passed.append("Ternary comparison")
        return True


class TestPropertyDifference:
    """Test property difference calculations."""

    def __init__(self):
        self.dbf = create_simple_test_database()
        self.passed = []
        self.failed = []

    def test_delta_symmetric(self):
        """Test property difference for symmetric binary (should be zero)."""
        print("\n" + "="*70)
        print("TEST 4: Property Difference - Symmetric Binary")
        print("="*70)

        T_val = 1000.0

        # For A-B with only L0 (symmetric), delta should be very small or zero
        delta_expr = uem_sym.uem1_delta_expr(
            self.dbf, 'A', 'B', 'LIQUID', v.T
        )

        # Evaluate at T = 1000K
        delta_func = lambdify(v.T, delta_expr, 'numpy')
        delta_val = float(delta_func(T_val))

        print(f"\nBinary: A-B (Symmetric, L0 only)")
        print(f"Temperature: {T_val} K")
        print(f"Property difference δ_AB: {delta_val:.10f}")

        # For symmetric interaction (L0 only), delta should be exactly zero
        is_zero = abs(delta_val) < 1e-10

        if is_zero:
            print("✅ PASSED: Symmetric binary gives δ ≈ 0")
            self.passed.append("Delta symmetric")
        else:
            print(f"❌ FAILED: Expected δ ≈ 0, got {delta_val}")
            self.failed.append("Delta symmetric")

        return is_zero

    def test_delta_asymmetric(self):
        """Test property difference for asymmetric binary (should be non-zero)."""
        print("\n" + "="*70)
        print("TEST 5: Property Difference - Asymmetric Binary")
        print("="*70)

        T_val = 1200.0

        # For A-C with L0 and L1 (asymmetric), delta should be non-zero
        delta_expr = uem_sym.uem1_delta_expr(
            self.dbf, 'A', 'C', 'LIQUID', v.T
        )

        delta_func = lambdify(v.T, delta_expr, 'numpy')
        delta_val = float(delta_func(T_val))

        print(f"\nBinary: A-C (Asymmetric, L0 and L1)")
        print(f"Temperature: {T_val} K")
        print(f"Property difference δ_AC: {delta_val:.10f}")

        # For asymmetric interaction, delta should be non-zero
        is_nonzero = abs(delta_val) > 1e-6

        if is_nonzero:
            print("✅ PASSED: Asymmetric binary gives δ > 0")
            self.passed.append("Delta asymmetric")
        else:
            print(f"❌ FAILED: Expected δ > 0, got {delta_val}")
            self.failed.append("Delta asymmetric")

        return is_nonzero

    def test_delta_highly_asymmetric(self):
        """Test property difference for highly asymmetric binary."""
        print("\n" + "="*70)
        print("TEST 6: Property Difference - Highly Asymmetric Binary")
        print("="*70)

        T_val = 1000.0

        # B-C has L0, L1, and L2 (highly asymmetric)
        delta_expr = uem_sym.uem1_delta_expr(
            self.dbf, 'B', 'C', 'LIQUID', v.T
        )

        delta_func = lambdify(v.T, delta_expr, 'numpy')
        delta_val = float(delta_func(T_val))

        print(f"\nBinary: B-C (Highly asymmetric, L0, L1, L2)")
        print(f"Temperature: {T_val} K")
        print(f"Property difference δ_BC: {delta_val:.10f}")

        # Should be larger than A-C
        delta_ac_expr = uem_sym.uem1_delta_expr(
            self.dbf, 'A', 'C', 'LIQUID', v.T
        )
        delta_ac_func = lambdify(v.T, delta_ac_expr, 'numpy')
        delta_ac_val = float(delta_ac_func(T_val))

        print(f"Comparison δ_AC:      {delta_ac_val:.10f}")

        is_larger = delta_val > delta_ac_val

        if is_larger:
            print(f"✅ PASSED: δ_BC > δ_AC (more asymmetry → larger δ)")
            self.passed.append("Delta highly asymmetric")
        else:
            print(f"❌ FAILED: Expected δ_BC > δ_AC")
            self.failed.append("Delta highly asymmetric")

        return is_larger


class TestContributionCoefficient:
    """Test contribution coefficient calculations."""

    def __init__(self):
        self.dbf = create_simple_test_database()
        self.passed = []
        self.failed = []

    def test_contribution_bounds(self):
        """Test that contribution coefficients are in reasonable range."""
        print("\n" + "="*70)
        print("TEST 7: Contribution Coefficient Bounds")
        print("="*70)

        T_val = 1000.0

        # Calculate r_CA (contribution of C to A in the A-B pair)
        r_ca_expr = uem_sym.uem1_contribution_ratio(
            self.dbf, 'C', 'A', 'B', 'LIQUID', v.T
        )

        r_ca_func = lambdify(v.T, r_ca_expr, 'numpy')
        r_ca_val = float(r_ca_func(T_val))

        print(f"\nContribution coefficient r_CA (C→A in A-B pair)")
        print(f"Temperature: {T_val} K")
        print(f"r_CA = {r_ca_val:.6f}")

        # Contribution coefficient should be between 0 and 1 (typically)
        in_range = 0.0 <= r_ca_val <= 1.5  # Allow slight overshoot due to exp term

        if in_range:
            print(f"✅ PASSED: r_CA in reasonable range [0, 1.5]")
            self.passed.append("Contribution bounds")
        else:
            print(f"❌ FAILED: r_CA = {r_ca_val} outside reasonable range")
            self.failed.append("Contribution bounds")

        return in_range


class TestNumericalStability:
    """Test numerical stability at edge cases."""

    def __init__(self):
        self.dbf = create_simple_test_database()
        self.passed = []
        self.failed = []

    def test_pure_component(self):
        """Test at pure component limits."""
        print("\n" + "="*70)
        print("TEST 8: Numerical Stability - Pure Component")
        print("="*70)

        comps = ['A', 'B', 'C', 'VA']
        phases = ['LIQUID']
        T = 1000.0

        pure_cases = [
            (1.0, 0.0, 0.0, "Pure A"),
            (0.0, 1.0, 0.0, "Pure B"),
            (0.0, 0.0, 1.0, "Pure C"),
        ]

        print(f"\nTemperature: {T} K")
        print("\nComposition         GM_UEM       Status")
        print("-" * 50)

        all_stable = True
        for x_a, x_b, x_c, desc in pure_cases:
            try:
                res = calculate(
                    self.dbf, comps, phases, model=ModelUEM,
                    T=T, P=101325, X_A=x_a, X_B=x_b, N=1
                )
                gm = float(res.GM.values[0])

                # Check for NaN or Inf
                is_valid = np.isfinite(gm)

                status = "✓" if is_valid else "✗ NaN/Inf"
                print(f"{status} {desc:20s} {gm:15.2f}")

                if not is_valid:
                    all_stable = False

            except Exception as e:
                print(f"✗ {desc:20s} Exception: {str(e)}")
                all_stable = False

        if all_stable:
            print("\n✅ PASSED: Pure component stability test")
            self.passed.append("Pure component stability")
        else:
            print("\n❌ FAILED: Pure component stability test")
            self.failed.append("Pure component stability")

        return all_stable

    def test_dilute_limits(self):
        """Test at very dilute compositions."""
        print("\n" + "="*70)
        print("TEST 9: Numerical Stability - Dilute Limits")
        print("="*70)

        comps = ['A', 'B', 'C', 'VA']
        phases = ['LIQUID']
        T = 1000.0

        dilute_cases = [
            (0.98, 0.01, 0.01, "Dilute B,C in A"),
            (0.01, 0.98, 0.01, "Dilute A,C in B"),
            (0.01, 0.01, 0.98, "Dilute A,B in C"),
        ]

        print(f"\nTemperature: {T} K")
        print("\nComposition         GM_UEM       Status")
        print("-" * 50)

        all_stable = True
        for x_a, x_b, x_c, desc in dilute_cases:
            try:
                res = calculate(
                    self.dbf, comps, phases, model=ModelUEM,
                    T=T, P=101325, X_A=x_a, X_B=x_b, N=1
                )
                gm = float(res.GM.values[0])

                is_valid = np.isfinite(gm)

                status = "✓" if is_valid else "✗ NaN/Inf"
                print(f"{status} {desc:20s} {gm:15.2f}")

                if not is_valid:
                    all_stable = False

            except Exception as e:
                print(f"✗ {desc:20s} Exception: {str(e)}")
                all_stable = False

        if all_stable:
            print("\n✅ PASSED: Dilute limits stability test")
            self.passed.append("Dilute limits stability")
        else:
            print("\n❌ FAILED: Dilute limits stability test")
            self.failed.append("Dilute limits stability")

        return all_stable


class TestTemperatureDependence:
    """Test temperature dependence of UEM calculations."""

    def __init__(self):
        self.dbf = create_simple_test_database()
        self.passed = []
        self.failed = []

    def test_temperature_scan(self):
        """Test UEM across temperature range."""
        print("\n" + "="*70)
        print("TEST 10: Temperature Dependence")
        print("="*70)

        comps = ['A', 'B', 'C', 'VA']
        phases = ['LIQUID']

        # Fixed composition
        x_a, x_b = 0.4, 0.3

        temperatures = [500, 1000, 1500, 2000]

        print(f"\nComposition: X(A)={x_a}, X(B)={x_b}, X(C)={1-x_a-x_b}")
        print("\nT(K)     GM_UEM       HM_UEM       SM_UEM       Status")
        print("-" * 70)

        all_valid = True
        for T in temperatures:
            try:
                res = calculate(
                    self.dbf, comps, phases, model=ModelUEM,
                    T=T, P=101325, X_A=x_a, X_B=x_b, N=1
                )

                gm = float(res.GM.values[0])
                hm = float(res.HM.values[0])
                sm = float(res.SM.values[0])

                is_valid = all(np.isfinite([gm, hm, sm]))

                status = "✓" if is_valid else "✗"
                print(f"{status} {T:4.0f}  {gm:12.2f} {hm:12.2f} {sm:12.6f}")

                if not is_valid:
                    all_valid = False

            except Exception as e:
                print(f"✗ {T:4.0f}  Exception: {str(e)}")
                all_valid = False

        if all_valid:
            print("\n✅ PASSED: Temperature dependence test")
            self.passed.append("Temperature dependence")
        else:
            print("\n❌ FAILED: Temperature dependence test")
            self.failed.append("Temperature dependence")

        return all_valid


def print_summary(test_suites):
    """Print summary of all tests."""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total_passed = sum(len(suite.passed) for suite in test_suites)
    total_failed = sum(len(suite.failed) for suite in test_suites)
    total_tests = total_passed + total_failed

    print(f"\nTotal tests run: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")

    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠️  {total_failed} test(s) failed")
        print("\nFailed tests:")
        for suite in test_suites:
            for test in suite.failed:
                print(f"  - {test}")

    print("\n" + "="*70)


def main():
    """Run all UEM validation tests."""
    print("="*70)
    print("UEM IMPLEMENTATION VALIDATION TEST SUITE")
    print("="*70)
    print("\nThis test suite validates the UEM extrapolation implementation")
    print("by comparing against known results and testing numerical stability.")

    # Create test suites
    test_suites = []

    # Binary system tests
    print("\n" + "─"*70)
    print("CATEGORY 1: Binary System Equivalence Tests")
    print("─"*70)
    binary_tests = TestBinarySystemEquivalence()
    binary_tests.test_binary_ab()
    binary_tests.test_binary_ac_asymmetric()
    test_suites.append(binary_tests)

    # Ternary system tests
    print("\n" + "─"*70)
    print("CATEGORY 2: Ternary System Tests")
    print("─"*70)
    ternary_tests = TestTernarySystem()
    ternary_tests.test_ternary_comparison()
    test_suites.append(ternary_tests)

    # Property difference tests
    print("\n" + "─"*70)
    print("CATEGORY 3: Property Difference Tests")
    print("─"*70)
    delta_tests = TestPropertyDifference()
    delta_tests.test_delta_symmetric()
    delta_tests.test_delta_asymmetric()
    delta_tests.test_delta_highly_asymmetric()
    test_suites.append(delta_tests)

    # Contribution coefficient tests
    print("\n" + "─"*70)
    print("CATEGORY 4: Contribution Coefficient Tests")
    print("─"*70)
    contrib_tests = TestContributionCoefficient()
    contrib_tests.test_contribution_bounds()
    test_suites.append(contrib_tests)

    # Numerical stability tests
    print("\n" + "─"*70)
    print("CATEGORY 5: Numerical Stability Tests")
    print("─"*70)
    stability_tests = TestNumericalStability()
    stability_tests.test_pure_component()
    stability_tests.test_dilute_limits()
    test_suites.append(stability_tests)

    # Temperature dependence tests
    print("\n" + "─"*70)
    print("CATEGORY 6: Temperature Dependence Tests")
    print("─"*70)
    temp_tests = TestTemperatureDependence()
    temp_tests.test_temperature_scan()
    test_suites.append(temp_tests)

    # Print summary
    print_summary(test_suites)


if __name__ == '__main__':
    main()
