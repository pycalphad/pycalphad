# Redlich-Kister-UEM: Understanding the CALPHAD Hierarchy

## Quick Reference

### The Two-Step CALPHAD Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    CALPHAD Modeling Hierarchy                    │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Binary Interaction Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Method:    Redlich-Kister Polynomials
  Formula:   G_ex^ij = x_i·x_j · Σ_n L^n_ij · (x_i - x_j)^n
  Status:    SAME for ALL methods (Muggianu, Kohler, Toop, UEM)

  Example Binary Parameters:
    AL-NI: L⁰ = -162407.75 + 16.212965·T
           L¹ = +73417.798 - 34.914168·T
           L² = +33471.014 - 9.8373558·T


STEP 2: Multicomponent Extrapolation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Status:    DIFFERENT methods

  ┌─────────────────────────────────────────────────────┐
  │  Redlich-Kister-Muggianu (Traditional)              │
  ├─────────────────────────────────────────────────────┤
  │  • Geometric symmetric averaging                    │
  │  • G_ex = Σ_{i<j} w_ij · G_ex^ij(x_i, x_j)         │
  │  • Weight: w_ij = 2·x_i·x_j / (x_i + x_j)          │
  │  • Fast and simple                                  │
  │  • No consideration of component similarity         │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  Redlich-Kister-UEM (This Implementation)           │
  ├─────────────────────────────────────────────────────┤
  │  • Property-difference-based extrapolation          │
  │  • Uses effective mole fractions                    │
  │  • x_eff_i = x_i + Σ_k r_ki·x_k                    │
  │  • Contribution coefficient r based on δ            │
  │  • δ_ij from binary Redlich-Kister slopes          │
  │  • Accounts for component similarity                │
  └─────────────────────────────────────────────────────┘
```

## Terminology Guide

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
