# Testing the UEM Implementation

This guide explains how to test and validate the UEM (Unified Extrapolation Model) implementation in pycalphad.

## Quick Start

### Running the Validation Test Suite

```bash
cd /path/to/pycalphad
python test_uem_validation.py
```

This will run a comprehensive test suite that validates:
- Binary system equivalence (UEM = Standard for binaries)
- Ternary system calculations (UEM vs Muggianu)
- Property difference calculations
- Contribution coefficients
- Numerical stability
- Temperature dependence

Expected runtime: ~10-30 seconds

### Running Unit Tests

```bash
pytest pycalphad/tests/test_model_uem.py -v
```

This runs the unit test suite with detailed test coverage.

## Test Files Overview

### 1. `test_uem_validation.py` (Root Directory)

**Purpose:** Comprehensive numerical validation with known test cases

**What it tests:**
- ✅ Binary systems: UEM should give identical results to standard Redlich-Kister
- ✅ Ternary systems: UEM should differ from Muggianu (demonstrates extrapolation)
- ✅ Property differences: Validates δ calculations for symmetric/asymmetric binaries
- ✅ Contribution coefficients: Validates r_ki bounds and calculations
- ✅ Numerical stability: Tests pure components and dilute limits
- ✅ Temperature dependence: Validates across 500-2000K range

**Test Database:**
Creates a simple A-B-C ternary system with:
- A-B: Symmetric (L⁰ = -10000)
- A-C: Asymmetric (L⁰ = -15000+5T, L¹ = -3000)
- B-C: Highly asymmetric (L⁰ = -20000, L¹ = -5000, L² = -2000)

**Example Output:**
```
TEST 1: Binary System A-B Equivalence
X(A)    GM_Standard      GM_UEM           Difference
----------------------------------------------------------------------
✓ 0.1      -123.456789    -123.456789    0.000000001
✓ 0.3      -456.789012    -456.789012    0.000000000
✓ 0.5      -567.890123    -567.890123    0.000000002

✅ PASSED: Binary A-B equivalence test
```

### 2. `pycalphad/tests/test_model_uem.py`

**Purpose:** Unit tests for model components and integration

**What it tests:**
- Model instantiation
- Contribution definitions
- Binary parameter handling
- Multicomponent system builds
- Integration with pycalphad calculate/equilibrium
- Real system tests (Al-Cr-Ni)

**Usage:**
```bash
pytest pycalphad/tests/test_model_uem.py::TestUEMBasic -v
pytest pycalphad/tests/test_model_uem.py::TestUEMTernary -v
pytest pycalphad/tests/test_model_uem.py -k "binary" -v
```

## Understanding Test Results

### Binary System Tests

**Expected:** UEM = Standard Model (within numerical tolerance)

**Why?** For binary systems, no multicomponent extrapolation is needed:
- Both use same Redlich-Kister polynomials
- No third component to create effective mole fractions
- Results should be numerically identical

**Tolerance:** < 1e-6 J/mol (1 microjoule)

**If this fails:**
- Check UEM implementation
- Verify binary parameter reading
- Ensure normalization is correct

### Ternary System Tests

**Expected:** UEM ≠ Muggianu (differences are normal and expected)

**Why?** Different extrapolation methods:
- Muggianu: Geometric symmetric averaging
- UEM: Property-difference-based effective mole fractions

**Typical differences:**
- Symmetric systems: 1-5% difference
- Asymmetric systems: 5-15% difference
- Highly asymmetric: Can be >15%

**If differences are extremely large (>50%):**
- Check for implementation errors
- Verify property difference calculations
- Check contribution coefficient calculations

### Property Difference Tests

**Expected:**
- Symmetric binary (L⁰ only): δ ≈ 0 (< 1e-10)
- Asymmetric binary (L⁰, L¹): δ > 0
- Highly asymmetric (L⁰, L¹, L²): δ >> 0

**Physical meaning:**
- δ = 0: Components behave identically
- Small δ: Similar components
- Large δ: Dissimilar components

**Calculation:**
```
δ_ij = |∂G_ex/∂x|_{x=0} - |∂G_ex/∂x|_{x=1}| / (R·T)
```

### Contribution Coefficient Tests

**Expected:** r_ki typically in [0, 1.5]

**Physical meaning:**
- r ≈ 0: Component k is very different from i
- r ≈ 0.5: Intermediate similarity
- r ≈ 1: Component k is similar to i

**Formula:**
```
r_ki = (δ_kj / (δ_ki + δ_kj)) * exp(-δ_ki)
```

### Numerical Stability Tests

**Expected:** All results finite (no NaN, no Inf)

**Critical cases:**
- Pure components: x_i = 1, x_j = 0
- Dilute limits: x_i ≈ 0
- Division by zero protection
- Logarithmic singularities

**If failures occur:**
- Check Piecewise definitions
- Verify division by zero handling
- Examine edge case logic

## Creating Custom Tests

### Testing with Your Own Database

```python
from pycalphad import Database, calculate
from pycalphad.models.model_uem import ModelUEM

# Load your database
dbf = Database('your_database.tdb')

# Define system
comps = ['AL', 'CR', 'NI', 'VA']
phases = ['LIQUID']

# Test at specific composition
conditions = {
    v.T: 1800,
    v.P: 101325,
    v.X('AL'): 0.33,
    v.X('CR'): 0.33,
    v.N: 1.0
}

# Calculate with both models
result_muggianu = calculate(dbf, comps, phases, conditions)
result_uem = calculate(dbf, comps, phases, model=ModelUEM, conditions=conditions)

# Compare
print(f"Muggianu: {result_muggianu.GM.values}")
print(f"UEM:      {result_uem.GM.values}")
print(f"Diff:     {result_uem.GM.values - result_muggianu.GM.values}")
```

### Testing Property Differences

```python
from pycalphad import Database, variables as v
from pycalphad.models.uem_symbolic import uem1_delta_expr
from sympy import lambdify

dbf = Database('your_database.tdb')

# Calculate property difference
delta_expr = uem1_delta_expr(dbf, 'AL', 'NI', 'LIQUID', v.T)

# Evaluate at temperature
delta_func = lambdify(v.T, delta_expr, 'numpy')
delta_value = delta_func(1800)  # At 1800K

print(f"Property difference δ_AL-NI at 1800K: {delta_value}")
```

### Testing Across Composition Space

```python
import numpy as np
import matplotlib.pyplot as plt

# Create composition grid
n_points = 20
x_vals = np.linspace(0.1, 0.9, n_points)

gm_muggianu = []
gm_uem = []

for x_al in x_vals:
    # Calculate with both models
    res_mug = calculate(dbf, comps, phases, T=1800, P=101325, X_AL=x_al, N=1)
    res_uem = calculate(dbf, comps, phases, model=ModelUEM, T=1800, P=101325, X_AL=x_al, N=1)

    gm_muggianu.append(res_mug.GM.values[0])
    gm_uem.append(res_uem.GM.values[0])

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(x_vals, gm_muggianu, 'b-', label='Muggianu')
plt.plot(x_vals, gm_uem, 'r--', label='UEM')
plt.xlabel('X(AL)')
plt.ylabel('G_m (J/mol)')
plt.legend()
plt.title('UEM vs Muggianu Extrapolation')
plt.grid(True)
plt.show()
```

## Interpreting Differences

### When should UEM and Muggianu give similar results?

1. **Binary systems:** Always identical
2. **Symmetric ternary:** Small differences (1-5%)
3. **Dilute solutions:** Small differences
4. **Components with similar properties:** Moderate differences

### When should UEM and Muggianu differ significantly?

1. **Highly asymmetric binaries:** Large L¹, L² parameters
2. **Dissimilar components:** Large property differences δ
3. **Multicomponent systems:** 4+ components
4. **Strong composition dependence:** Non-linear binary interactions

### Which is "correct"?

Neither is universally correct! Both are extrapolation approximations:

- **Muggianu:** Geometric averaging, well-established, computationally fast
- **UEM:** Property-based, physically motivated, better for asymmetric systems

**Best practice:**
1. Calculate with both methods
2. Compare results to assess extrapolation uncertainty
3. Validate against experimental data when available
4. For critical applications, measure ternary data directly

## Troubleshooting

### Test Failures

**"Binary equivalence test failed"**
- Check UEM implementation in `uem_symbolic.py`
- Verify normalization in `model_uem.py`
- Ensure no ternary parameters are being used

**"Numerical stability test failed"**
- Check edge case handling (x=0, x=1)
- Verify Piecewise expressions
- Look for division by zero issues

**"Property difference test failed"**
- Check derivative calculations in `uem1_delta_expr`
- Verify symbolic simplification
- Ensure proper absolute value handling

### Performance Issues

**Tests running slowly**
- Normal for first run (symbolic compilation)
- Subsequent runs should be faster
- Large systems (5+ components) naturally slower

### Installation Issues

**Import errors**
```
ModuleNotFoundError: No module named 'pycalphad.models.model_uem'
```

Solution:
1. Ensure you're in the correct directory
2. Install pycalphad in development mode:
   ```bash
   pip install -e .
   ```

## Continuous Integration

For automated testing in CI/CD pipelines:

```bash
# Run all UEM tests
python test_uem_validation.py && pytest pycalphad/tests/test_model_uem.py -v

# Check for failures
if [ $? -eq 0 ]; then
    echo "All UEM tests passed"
else
    echo "UEM tests failed"
    exit 1
fi
```

## Reporting Issues

If tests fail unexpectedly:

1. **Gather information:**
   - Test output
   - Python version
   - pycalphad version
   - SymPy/SymEngine version

2. **Minimal example:**
   - Simplest code that reproduces the issue
   - Sample database if needed

3. **Report:**
   - GitHub Issues: https://github.com/pycalphad/pycalphad/issues
   - Include test output and versions

## References

- Chou, K. C. (2020). Fluid Phase Equilibria, 507, 112416.
- Chou, K. C., Wei, S. K. (2020). J. Molecular Liquids, 298, 111951.
- Chou, K. C. et al. (2024). Thermochimica Acta, 179824.

---

Last updated: 2025-10-21
