"""
Redlich-Kister-UEM Usage Examples

This script demonstrates how to use the UEM extrapolation method in pycalphad for
calculating thermodynamic properties of multicomponent solution phases.

IMPORTANT TERMINOLOGY:
---------------------
- Binary systems: Both methods use Redlich-Kister polynomials (IDENTICAL)
- Multicomponent: UEM vs Muggianu extrapolation (DIFFERENT)
- Correct naming: "Redlich-Kister-UEM" vs "Redlich-Kister-Muggianu"
- NOT: "UEM" vs "Redlich-Kister"

The UEM extrapolation method provides an alternative to traditional Muggianu/Kohler/Toop
geometric extrapolation by using property-difference-based effective mole fractions.
"""

import numpy as np
from pycalphad import Database, calculate, equilibrium, variables as v
from pycalphad.models.model_uem import ModelUEM
from pycalphad import Model
import matplotlib.pyplot as plt


def example_1_binary_comparison():
    """
    Example 1: Compare Redlich-Kister-UEM with Redlich-Kister-Muggianu for binary system.

    For binary systems, both methods give IDENTICAL results because:
    - Both use the same Redlich-Kister polynomials
    - No multicomponent extrapolation is needed (only 2 components)
    - This verifies correct implementation
    """
    print("=" * 70)
    print("Example 1: Binary System - Redlich-Kister (UEM vs Muggianu)")
    print("=" * 70)

    # Load database
    dbf = Database('alcrni.tdb')

    # Binary system: AL-NI
    comps = ['AL', 'NI', 'VA']
    phases = ['LIQUID']

    # Conditions
    conditions = {
        v.T: 1800,
        v.P: 101325,
        v.X('AL'): (0, 1, 0.1)  # Scan from 0 to 1
    }

    # Calculate with standard model
    print("\nCalculating with standard Redlich-Kister model...")
    result_standard = calculate(dbf, comps, phases, conditions)

    # Calculate with UEM model
    print("Calculating with UEM model...")
    result_uem = calculate(dbf, comps, phases, model=ModelUEM, conditions=conditions)

    # Compare results
    print("\nResults at X(AL) = 0.5:")
    idx = np.argmin(np.abs(result_standard.X_AL.values - 0.5))
    print(f"  Standard model GM: {result_standard.GM.values[0, idx]:.2f} J/mol")
    print(f"  UEM model GM:      {result_uem.GM.values[0, idx]:.2f} J/mol")
    print(f"  Difference:        {abs(result_standard.GM.values[0, idx] - result_uem.GM.values[0, idx]):.2f} J/mol")

    return result_standard, result_uem


def example_2_ternary_system():
    """
    Example 2: Ternary system calculation with UEM.

    This demonstrates UEM extrapolation from binary to ternary systems,
    where differences from standard models become apparent.
    """
    print("\n" + "=" * 70)
    print("Example 2: Ternary System - AL-CR-NI LIQUID")
    print("=" * 70)

    # Load database
    dbf = Database('alcrni.tdb')

    # Ternary system
    comps = ['AL', 'CR', 'NI', 'VA']
    phases = ['LIQUID']

    # Fixed composition point
    conditions = {
        v.T: 1800,
        v.P: 101325,
        v.X('AL'): 0.33,
        v.X('CR'): 0.33,
        # X(NI) = 1 - X(AL) - X(CR) = 0.34
        v.N: 1.0
    }

    # Calculate with standard model (Muggianu)
    print("\nCalculating with standard Muggianu model...")
    result_standard = calculate(dbf, comps, phases, conditions)

    # Calculate with UEM model
    print("Calculating with UEM model...")
    result_uem = calculate(dbf, comps, phases, model=ModelUEM, conditions=conditions)

    # Report results
    print("\nResults at equimolar composition:")
    print(f"  Standard model (Muggianu):")
    print(f"    GM:   {result_standard.GM.values[0]:.2f} J/mol")
    print(f"    HM:   {result_standard.HM.values[0]:.2f} J/mol")
    print(f"    SM:   {result_standard.SM.values[0]:.4f} J/mol-K")

    print(f"\n  UEM model:")
    print(f"    GM:   {result_uem.GM.values[0]:.2f} J/mol")
    print(f"    HM:   {result_uem.HM.values[0]:.2f} J/mol")
    print(f"    SM:   {result_uem.SM.values[0]:.4f} J/mol-K")

    print(f"\n  Differences:")
    print(f"    ΔGM:  {result_uem.GM.values[0] - result_standard.GM.values[0]:.2f} J/mol")
    print(f"    ΔHM:  {result_uem.HM.values[0] - result_standard.HM.values[0]:.2f} J/mol")

    return result_standard, result_uem


def example_3_composition_scan():
    """
    Example 3: Scan across composition space in ternary system.

    Compare UEM and standard model predictions across a range of compositions.
    """
    print("\n" + "=" * 70)
    print("Example 3: Composition Scan - Comparing UEM vs Muggianu")
    print("=" * 70)

    # Load database
    dbf = Database('alcrni.tdb')
    comps = ['AL', 'CR', 'NI', 'VA']
    phases = ['LIQUID']

    # Create composition grid
    n_points = 5
    x_al_values = np.linspace(0.1, 0.8, n_points)
    x_cr_values = np.linspace(0.1, 0.8, n_points)

    print(f"\nScanning {n_points}x{n_points} composition grid...")
    print("X(AL)   X(CR)   X(NI)   | GM_Std       GM_UEM       Difference")
    print("-" * 70)

    results = []

    for x_al in x_al_values:
        for x_cr in x_cr_values:
            x_ni = 1.0 - x_al - x_cr
            if x_ni < 0.05:  # Skip invalid compositions
                continue

            conditions = {
                v.T: 1800,
                v.P: 101325,
                v.X('AL'): x_al,
                v.X('CR'): x_cr,
                v.N: 1.0
            }

            try:
                # Calculate both models
                res_std = calculate(dbf, comps, phases, conditions)
                res_uem = calculate(dbf, comps, phases, model=ModelUEM, conditions=conditions)

                gm_std = res_std.GM.values[0]
                gm_uem = res_uem.GM.values[0]
                diff = gm_uem - gm_std

                print(f"{x_al:.2f}    {x_cr:.2f}    {x_ni:.2f}   | "
                      f"{gm_std:12.2f} {gm_uem:12.2f} {diff:12.2f}")

                results.append({
                    'X_AL': x_al,
                    'X_CR': x_cr,
                    'X_NI': x_ni,
                    'GM_std': gm_std,
                    'GM_uem': gm_uem,
                    'diff': diff
                })

            except Exception as e:
                print(f"{x_al:.2f}    {x_cr:.2f}    {x_ni:.2f}   | Error: {str(e)}")

    return results


def example_4_equilibrium_calculation():
    """
    Example 4: Phase equilibrium calculation with UEM.

    Demonstrates using UEM in equilibrium calculations to find stable phases
    and their compositions.
    """
    print("\n" + "=" * 70)
    print("Example 4: Equilibrium Calculation with UEM")
    print("=" * 70)

    # Load database
    dbf = Database('alcrni.tdb')

    # System setup
    comps = ['AL', 'CR', 'NI', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'BCC_A2']

    # Equilibrium conditions
    conditions = {
        v.T: 1400,
        v.P: 101325,
        v.X('AL'): 0.25,
        v.X('CR'): 0.25,
        v.N: 1.0
    }

    print("\nCalculating equilibrium at T=1400K, X(AL)=0.25, X(CR)=0.25...")

    # Calculate with standard model
    print("\nWith standard model:")
    try:
        eq_std = equilibrium(dbf, comps, phases, conditions)
        print(f"  Stable phases: {eq_std.Phase.values}")
        print(f"  Total Gibbs energy: {eq_std.GM.values[0]:.2f} J/mol")
    except Exception as e:
        print(f"  Error: {str(e)}")

    # Calculate with UEM model
    print("\nWith UEM model:")
    try:
        eq_uem = equilibrium(dbf, comps, phases, conditions, model=ModelUEM)
        print(f"  Stable phases: {eq_uem.Phase.values}")
        print(f"  Total Gibbs energy: {eq_uem.GM.values[0]:.2f} J/mol")
    except Exception as e:
        print(f"  Error: {str(e)}")

    return eq_std, eq_uem


def example_5_temperature_scan():
    """
    Example 5: Temperature dependence with UEM.

    Scan temperature at fixed composition to see how predictions vary.
    """
    print("\n" + "=" * 70)
    print("Example 5: Temperature Scan at Fixed Composition")
    print("=" * 70)

    # Load database
    dbf = Database('alcrni.tdb')
    comps = ['AL', 'CR', 'NI', 'VA']
    phases = ['LIQUID']

    # Fixed composition
    x_al = 0.4
    x_cr = 0.3
    x_ni = 0.3

    # Temperature range
    temperatures = np.linspace(1000, 2000, 11)

    print(f"\nComposition: X(AL)={x_al}, X(CR)={x_cr}, X(NI)={x_ni}")
    print("\nT(K)    | GM_Standard   GM_UEM        Difference")
    print("-" * 60)

    for T in temperatures:
        conditions = {
            v.T: T,
            v.P: 101325,
            v.X('AL'): x_al,
            v.X('CR'): x_cr,
            v.N: 1.0
        }

        try:
            res_std = calculate(dbf, comps, phases, conditions)
            res_uem = calculate(dbf, comps, phases, model=ModelUEM, conditions=conditions)

            gm_std = res_std.GM.values[0]
            gm_uem = res_uem.GM.values[0]
            diff = gm_uem - gm_std

            print(f"{T:6.0f}  | {gm_std:13.2f} {gm_uem:13.2f} {diff:13.2f}")

        except Exception as e:
            print(f"{T:6.0f}  | Error: {str(e)}")


def example_6_model_comparison_summary():
    """
    Example 6: Summary comparison of different extrapolation methods.

    Shows when Redlich-Kister-UEM predictions differ most from Redlich-Kister-Muggianu.
    """
    print("\n" + "=" * 70)
    print("Example 6: Redlich-Kister-UEM vs Muggianu - Key Differences")
    print("=" * 70)

    print("""
    CALPHAD Two-Step Modeling Process:
    ==================================

    Step 1: Binary Interaction Description (SAME for all methods)
    --------------------------------------------------------------
    - Redlich-Kister polynomials: G_ex^ij = x_i*x_j * Σ L^n_ij*(x_i-x_j)^n
    - Parameters L^n_ij from binary phase diagram assessments
    - Both UEM and Muggianu use IDENTICAL binary descriptions

    Step 2: Multicomponent Extrapolation (DIFFERENT)
    ------------------------------------------------

    Redlich-Kister-Muggianu (Traditional):
    - Geometric symmetric averaging of binary contributions
    - G_ex = Σ_{i<j} (2*x_i*x_j/(x_i+x_j)) * G_ex^ij
    - Treats all components equally
    - Fast and simple

    Redlich-Kister-UEM (This Implementation):
    - Property-difference-based effective mole fractions
    - Components with similar properties contribute more to each other
    - Property difference (delta) calculated from binary Redlich-Kister slopes
    - Provides unified framework covering Muggianu/Kohler/Toop as special cases

    When Extrapolation Methods Differ Most:
    1. Systems with highly dissimilar components
    2. Asymmetric binary subsystems (large L1, L2 parameters)
    3. Multicomponent systems (4+ components)
    4. When binary parameters show large composition dependence

    Advantages of UEM:
    - More physical basis (uses property differences)
    - Better extrapolation for asymmetric systems
    - Smooth composition dependence
    - No arbitrary asymmetric parameter selection

    Usage recommendations:
    - Use UEM for complex alloy systems with dissimilar elements
    - Compare with standard models to assess prediction uncertainty
    - Validate against experimental data when available
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("UNIFIED EXTRAPOLATION MODEL (UEM) - USAGE EXAMPLES")
    print("=" * 70)

    print("""
    This script demonstrates the UEM implementation in pycalphad.
    Examples cover binary systems, ternary extrapolation, composition
    scans, equilibrium calculations, and temperature dependence.
    """)

    try:
        # Run examples
        example_1_binary_comparison()
        example_2_ternary_system()
        example_3_composition_scan()
        example_4_equilibrium_calculation()
        example_5_temperature_scan()
        example_6_model_comparison_summary()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nError: Database file not found - {str(e)}")
        print("Please ensure 'alcrni.tdb' is in the examples directory.")
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
