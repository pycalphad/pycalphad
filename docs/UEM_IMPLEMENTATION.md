# Unified Extrapolation Model (UEM) Implementation in Pycalphad

## Overview

The Unified Extrapolation Model (UEM) is now implemented in pycalphad for calculating thermodynamic properties of multicomponent solution phases. This document describes the implementation, usage, and theoretical background.

## What is UEM?

The UEM is a method for predicting thermodynamic properties of multicomponent systems from binary subsystem data. It addresses the fundamental challenge in CALPHAD modeling: how to extrapolate from well-established binary parameters to multicomponent systems without requiring ternary (or higher) interaction parameters.

### Key Concepts

1. **Property Difference (δ)**: A dimensionless quantity characterizing the dissimilarity between two components based on their binary interaction:
   ```
   δ_ij = |∂G_ex/∂x|_{x=0} - |∂G_ex/∂x|_{x=1}| / (R*T)
   ```

2. **Contribution Coefficient (r)**: Determines how much a third component k contributes to the effective mole fraction of component i in the i-j binary subsystem:
   ```
   r_ki = (δ_kj / (δ_ki + δ_kj)) * exp(-δ_ki)
   ```

3. **Effective Mole Fractions**: Account for contributions from all components based on similarity:
   ```
   x_eff_i = x_i + Σ_k r_ki * x_k  (for k ≠ i, j)
   ```

4. **Normalized Effective Mole Fractions**: Used in binary excess energy expressions:
   ```
   X_ij = x_eff_i / (x_eff_i + x_eff_j)
   ```

5. **Total Excess Gibbs Energy**: Sum of weighted binary contributions:
   ```
   G_ex = Σ_{i<j} [(x_i * x_j) / (X_ij * X_ji)] * G_ex_ij(X_ij, X_ji)
   ```

## Implementation Structure

### File Organization

```
pycalphad/
├── models/
│   ├── model_uem.py         # Main UEM model class
│   └── uem_symbolic.py      # Symbolic expression builders
├── tests/
│   └── test_model_uem.py    # Comprehensive test suite
└── examples/
    └── uem_example.py       # Usage examples
```

### Core Components

#### 1. `ModelUEM` Class (`model_uem.py`)

Main model class that extends the standard pycalphad `Model`:

```python
from pycalphad.models.model_uem import ModelUEM

# Use in calculations
result = calculate(dbf, comps, phases, model=ModelUEM, conditions)
```

Features:
- Inherits reference and ideal mixing energy from base Model
- Overrides excess mixing energy with UEM formulation
- Compatible with all pycalphad workflows
- Automatically handles vacancies (VA)

#### 2. UEM Symbolic Functions (`uem_symbolic.py`)

Core symbolic computation functions:

- `is_binary_in_phase()`: Helper to identify binary parameters
- `uem1_delta_expr()`: Calculate property difference δ
- `uem1_contribution_ratio()`: Calculate contribution coefficient r
- `construct_binary_excess()`: Build binary excess energy with effective mole fractions
- `get_uem1_excess_gibbs_expr()`: Main function constructing total excess energy
- `is_stable_expression()`: Numerical stability checker

## Usage

### Basic Usage

```python
from pycalphad import Database, calculate, variables as v
from pycalphad.models.model_uem import ModelUEM

# Load database
dbf = Database('mydb.tdb')

# Define system
comps = ['AL', 'CR', 'NI', 'VA']
phases = ['LIQUID']

# Set conditions
conditions = {
    v.T: 1800,
    v.P: 101325,
    v.X('AL'): 0.33,
    v.X('CR'): 0.33,
    v.N: 1.0
}

# Calculate with UEM model
result = calculate(dbf, comps, phases, model=ModelUEM, conditions=conditions)

print(result.GM.values)  # Gibbs energy
```

### Equilibrium Calculations

```python
from pycalphad import equilibrium

# Calculate phase equilibrium with UEM
eq_result = equilibrium(dbf, comps, phases, conditions, model=ModelUEM)

print(eq_result.Phase.values)  # Stable phases
print(eq_result.NP.values)     # Phase fractions
```

### Comparison with Standard Model

```python
from pycalphad import Model

# Standard Muggianu model
result_std = calculate(dbf, comps, phases, conditions)

# UEM model
result_uem = calculate(dbf, comps, phases, model=ModelUEM, conditions=conditions)

# Compare
print(f"Standard: {result_std.GM.values}")
print(f"UEM:      {result_uem.GM.values}")
print(f"Diff:     {result_uem.GM.values - result_std.GM.values}")
```

## Mathematical Formulation

### Algorithm Flow

For a multicomponent system with n components:

1. **Iterate over all binary pairs (i,j)**:
   - Total number of pairs: n(n-1)/2

2. **For each pair, calculate effective mole fractions**:
   ```
   x_eff_i = x_i + Σ_{k≠i,j} r_ki * x_k
   x_eff_j = x_j + Σ_{k≠i,j} r_kj * x_k
   ```

3. **Normalize to binary subsystem**:
   ```
   X_ij = x_eff_i / (x_eff_i + x_eff_j)
   X_ji = x_eff_j / (x_eff_i + x_eff_j)
   ```

4. **Construct binary excess with Redlich-Kister parameters**:
   ```
   G_ex_ij = X_ij * X_ji * Σ_n L^n_ij * (X_ij - X_ji)^n
   ```
   where L^n_ij are binary interaction parameters from database

5. **Weight binary contribution**:
   ```
   Weight = (x_i * x_j) / (X_ij * X_ji)
   ```

6. **Sum all weighted binary contributions**:
   ```
   G_ex_total = Σ_{i<j} Weight_{ij} * G_ex_ij
   ```

### Property Difference Calculation

The property difference δ_ij is calculated from the binary excess Gibbs energy:

```python
# Symbolic variable for composition
x = Symbol('x')

# Build Redlich-Kister expression
G_ex = x*(1-x) * Σ_n L_n * (2*x - 1)^n

# Derivatives at boundaries
dG_dx_0 = dG_ex/dx |_{x=0}
dG_dx_1 = dG_ex/dx |_{x=1}

# Property difference
δ = |dG_dx_0 - dG_dx_1| / (R*T)
```

Physical interpretation:
- δ = 0: Symmetric binary system (both components behave identically)
- δ > 0: Asymmetric system (components have different properties)
- Large δ: Very dissimilar components

### Contribution Coefficient

```python
r_ki = (δ_kj / (δ_ki + δ_kj)) * exp(-δ_ki)
```

Physical interpretation:
- If k is similar to i (small δ_ki): r_ki is larger (k contributes more to i)
- If k is similar to j (small δ_kj): r_ki is smaller (k contributes less to i)
- The exponential term exp(-δ_ki) enhances similarity effects

## Advantages and Limitations

### Advantages

1. **Physical Basis**: Uses property differences rather than arbitrary geometric rules
2. **Binary Parameters Only**: No need for ternary or higher-order parameters
3. **Smooth Extrapolation**: Provides continuous predictions across composition space
4. **Unified Framework**: Can reduce to Kohler, Muggianu, etc. as special cases
5. **Asymmetric Systems**: Better handles highly asymmetric component interactions

### Limitations

1. **Computational Cost**: More complex than simple geometric averaging
2. **Symbolic Complexity**: Expressions can become large for many components
3. **Database Requirements**: Requires well-characterized binary parameters
4. **Validation**: Should be compared with experimental data when available

### When to Use UEM

Use UEM when:
- Working with 3+ component systems
- Binary subsystems show significant asymmetry (large L1, L2 parameters)
- Components have very different chemical properties
- No ternary experimental data available for validation
- Exploring uncertainty in multicomponent extrapolation

Compare with standard models:
- Always validate against available experimental data
- Use both UEM and Muggianu to assess prediction uncertainty
- Consider the specific chemistry of your system

## Testing

Comprehensive test suite in `test_model_uem.py`:

```bash
pytest pycalphad/tests/test_model_uem.py -v
```

Test categories:
1. **Basic Infrastructure**: Model instantiation, contributions
2. **Binary Systems**: Verification against standard Redlich-Kister
3. **Ternary Systems**: Multicomponent extrapolation
4. **Numerical Stability**: Edge cases, pure components, equimolar
5. **Real Systems**: Al-Cr-Ni example database

## Examples

See `examples/uem_example.py` for detailed examples:

1. Binary comparison (UEM vs standard)
2. Ternary system calculation
3. Composition space scanning
4. Equilibrium calculations
5. Temperature dependence
6. Model comparison summary

Run examples:
```bash
python examples/uem_example.py
```

## References

1. Chou, K. C. (2020). "On the definition of the components' difference in properties in the unified extrapolation model." *Fluid Phase Equilibria*, 507, 112416.

2. Chou, K. C., Wei, S. K. (2020). "New expression for property difference in components for the Unified Extrapolation Model." *Journal of Molecular Liquids*, 298, 111951.

3. Chou, K. C. et al. (2024). "Latest formulations of the Unified Extrapolation Model." *Thermochimica Acta*, 179824.

4. Lukas, H. L., Fries, S. G., Sundman, B. (2007). *Computational Thermodynamics: The Calphad Method*. Cambridge University Press.

## Implementation Details

### Numerical Stability

Several measures ensure numerical stability:

1. **Division by Zero Protection**:
   ```python
   if Xi_ij == S.Zero or Xj_ij == S.Zero:
       ratio = S.Zero
   else:
       ratio = (x_i * x_j) / (Xi_ij * Xj_ij)
   ```

2. **NaN Handling**:
   ```python
   total_expr = total_expr.subs(nan, 0)
   ```

3. **Default Ratios**: When property differences are zero (identical components):
   ```python
   if delta_ki == S.Zero and delta_kj == S.Zero:
       return S.Half  # Equal contribution
   ```

4. **Stability Checker**: `is_stable_expression()` verifies no infinities or negative powers

### Performance Considerations

For n-component systems:
- Number of binary pairs: n(n-1)/2
- For each pair, calculate contributions from (n-2) other components
- Total operations scale as O(n³)

For 10-component system:
- 45 binary pairs
- 8 contributions per pair
- 360 total contribution calculations

Optimization strategies:
- Symbolic expressions are compiled once via SymEngine
- Vectorized numerical evaluation via NumPy
- Caching of intermediate results where possible

## Troubleshooting

### Common Issues

1. **Missing binary parameters**:
   - UEM treats missing binaries as ideal (G_ex = 0)
   - Warning logged: "No interaction parameters found for X-Y"
   - Solution: Ensure database has all binary parameters

2. **Numerical instability**:
   - May occur at composition boundaries
   - Check for very large or very small parameter values
   - Use stability checker: `is_stable_expression(expr)`

3. **Different results from standard model**:
   - Expected behavior for multicomponent systems
   - UEM uses different extrapolation method
   - Compare both; validate against experimental data

4. **Slow calculations**:
   - Normal for large systems (many components)
   - Consider reducing number of components if possible
   - Use equilibrium calculations rather than grid scanning

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show:
- Binary pairs being processed
- Contribution coefficients calculated
- Any warnings or errors

## Future Enhancements

Potential improvements:

1. **UEM2 Variant**: Alternative property difference definition
2. **Caching**: Memoize delta and contribution calculations
3. **Parallelization**: Compute binary pairs in parallel
4. **Symbolic Simplification**: More aggressive simplification
5. **Higher-Order Terms**: Optional ternary parameter support

## Contributing

To contribute to UEM development:

1. Fork the pycalphad repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all existing tests pass
5. Submit pull request with detailed description

## License

The UEM implementation is part of pycalphad and uses the same MIT license.

## Contact

For questions, issues, or contributions:
- GitHub Issues: https://github.com/pycalphad/pycalphad/issues
- Documentation: https://pycalphad.org
- Mailing List: https://groups.google.com/forum/#!forum/pycalphad

---

*Last updated: 2025-10-21*
