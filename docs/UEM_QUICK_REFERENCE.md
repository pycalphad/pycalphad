# Redlich-Kister-UEM: Understanding the CALPHAD Hierarchy

## Quick Reference

### The Two-Step CALPHAD Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    CALPHAD Modeling Hierarchy                    │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Binary Interaction Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Methods Available:
    • Redlich-Kister polynomials (most common)
    • MQMQA (Modified Quasichemical Model)
    • Associate solution models
    • Other thermodynamic models

  For Redlich-Kister:
    Formula:   G_ex^ij = x_i·x_j · Σ_n L^n_ij · (x_i - x_j)^n
    Status:    SAME for ALL extrapolation methods

  Example Binary Parameters:
    AL-NI: L⁰ = -162407.75 + 16.212965·T
           L¹ = +73417.798 - 34.914168·T
           L² = +33471.014 - 9.8373558·T


STEP 2: Multicomponent Extrapolation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Status:    DIFFERENT methods

  ┌─────────────────────────────────────────────────────┐
  │  Traditional Geometric Extrapolation                │
  ├─────────────────────────────────────────────────────┤
  │  • Muggianu (symmetric)                             │
  │  • Kohler (asymmetric)                              │
  │  • Toop (asymmetric with designated component)      │
  │  • G_ex = Σ_{i<j} w_ij · G_ex^ij(x_i, x_j)         │
  │  • Fast and simple                                  │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  UEM Extrapolation (This Implementation)            │
  ├─────────────────────────────────────────────────────┤
  │  • Property-difference-based extrapolation          │
  │  • Uses effective mole fractions                    │
  │  • x_eff_i = x_i + Σ_k r_ki·x_k                    │
  │  • Works with ANY binary model:                     │
  │    - Redlich-Kister-UEM                            │
  │    - MQMQA-UEM                                      │
  │    - Associate-UEM                                  │
  │  • Accounts for component similarity                │
  └─────────────────────────────────────────────────────┘

  KEY INSIGHT: UEM is a multicomponent EXTRAPOLATION method
               that works with different binary DESCRIPTION models
```

## Compatibility with Different Binary Models

### UEM as Universal Extrapolation Method

UEM is a **multicomponent extrapolation framework** that can work with various binary description models:

| Binary Model | Multicomponent Extrapolation | Full Name |
|-------------|------------------------------|-----------|
| Redlich-Kister | UEM | Redlich-Kister-UEM |
| MQMQA | UEM | MQMQA-UEM |
| Associate | UEM | Associate-UEM |
| Any model | UEM | [Model]-UEM |

### Example: MQMQA-UEM

```
Binary Description: MQMQA (Modified Quasichemical Model)
  → Describes short-range ordering in liquid
  → Uses quadruplet approximation
  → Particularly good for ionic and molten salt systems

Multicomponent Extrapolation: UEM
  → Property differences calculated from MQMQA binary data
  → Effective mole fractions based on MQMQA energetics
  → Same UEM framework, different binary input

Result: MQMQA-UEM
  → Better binary description for certain systems (e.g., molten salts)
  → Better multicomponent extrapolation than geometric averaging
```

### Implementation Status

**Currently Implemented:**
- ✅ Redlich-Kister-UEM (this module)

**Possible Future Extensions:**
- ⏳ MQMQA-UEM (combines ModelMQMQA + UEM extrapolation)
- ⏳ Associate-UEM
- ⏳ Generic UEM framework for any binary model

### Key Principle

```
UEM Extrapolation = f(Binary_Energy_Function)

Where Binary_Energy_Function can be:
  • Redlich-Kister polynomial
  • MQMQA energy
  • Associate solution energy
  • Any differentiable binary excess energy
```

The property difference δ is always calculated from:
```
δ_ij = |∂G_ex^ij/∂x|_{x=0} - |∂G_ex^ij/∂x|_{x=1}| / (R·T)
```

Regardless of what model provides G_ex^ij!

### ✅ CORRECT Usage

| Context | Correct Term |
|---------|--------------|
| Binary systems | Redlich-Kister polynomials |
| Traditional ternary+ | Redlich-Kister-Muggianu |
| Alternative ternary+ | Redlich-Kister-UEM |
| Comparison | "RK-UEM vs RK-Muggianu" |
| Short form | "UEM extrapolation" |

### ❌ INCORRECT Usage (Common Mistakes)

| Incorrect | Why Wrong | Should Say |
|-----------|-----------|------------|
| "UEM vs Redlich-Kister" | Implies different binary description | "RK-UEM vs RK-Muggianu" |
| "UEM model" | Vague, suggests complete model | "UEM extrapolation method" |
| "UEM replaces Redlich-Kister" | False, uses same RK binaries | "UEM is alternative extrapolation" |
| "New model for binaries" | Wrong, binaries unchanged | "New extrapolation for ternary+" |

## When Methods Give Same vs Different Results

### Identical Results
- **Binary systems (2 components)**
  - Both use Redlich-Kister polynomials
  - No extrapolation needed
  - Results are numerically identical

### Different Results
- **Ternary systems (3 components)**
  - Modest differences
  - Depends on asymmetry of binaries

- **Quaternary+ systems (4+ components)**
  - Larger differences
  - UEM accounts for more complex interactions

## Example Comparison

For **AL-CR-NI** ternary liquid at 1800K, X_AL=0.33, X_CR=0.33:

```
Method                    G_excess (J/mol)    Notes
─────────────────────────────────────────────────────────
RK-Muggianu              -25,431             Geometric avg
RK-UEM                   -26,108             Property-based
Difference                  -677             ~2.7% difference
```

The difference arises because:
- AL, CR, NI have different binary interaction strengths
- UEM weights contributions based on component similarity
- Muggianu treats all pairs equally

## Understanding Property Difference (δ)

The property difference δ_ij is calculated from binary Redlich-Kister data:

```python
# From binary AL-NI with parameters L0, L1, L2, ...
# Build: G_ex = x*(1-x) * [L0 + L1*(2x-1) + L2*(2x-1)² + ...]

# Calculate slopes at boundaries
slope_at_pure_AL  = dG_ex/dx |_{x→1}
slope_at_pure_NI  = dG_ex/dx |_{x→0}

# Property difference (dimensionless)
δ_AL-NI = |slope_at_pure_AL - slope_at_pure_NI| / (R·T)
```

Physical meaning:
- **δ = 0**: Symmetric binary (like Raoult's law)
- **δ > 0**: Asymmetric binary (components behave differently)
- **Large δ**: Very dissimilar components

## Practical Implications

### For Database Developers
- Same binary assessments work for both methods
- No need to change existing TDB files
- Can compare extrapolation methods with same data

### For Users
- Try both methods for multicomponent systems
- Validate against experimental data when available
- For high asymmetry, UEM may be more accurate
- For simple systems, Muggianu is faster

### For Software Integration
- UEM is drop-in replacement: `model=ModelUEM`
- Same database format
- Same pycalphad workflow
- Only excess mixing energy calculation changes

## Summary

**What UEM Changes:**
- Multicomponent extrapolation method (Step 2)

**What UEM Does NOT Change:**
- Binary Redlich-Kister polynomials (Step 1)
- Database format
- Binary system calculations
- Reference or ideal mixing energies

**Result:**
- Same inputs (binary Redlich-Kister parameters)
- Different outputs (for ternary+ systems)
- Alternative predictions to assess uncertainty
