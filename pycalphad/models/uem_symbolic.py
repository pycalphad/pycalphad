"""
UEM (Unified Extrapolation Model) Symbolic Computation Module

This module implements the Redlich-Kister-UEM method for calculating thermodynamic
properties of multicomponent solution phases.

CALPHAD Modeling Hierarchy:
-------------------------
1. Binary systems: Described by Redlich-Kister polynomials
   G_ex^ij = x_i*x_j * Σ_n L^n_ij * (x_i - x_j)^n

2. Multicomponent extrapolation: Traditional vs UEM approaches
   - Traditional: Redlich-Kister-Muggianu (geometric symmetric averaging)
   - Traditional: Redlich-Kister-Kohler (geometric asymmetric averaging)
   - Traditional: Redlich-Kister-Toop (asymmetric with selected component)
   - This module: Redlich-Kister-UEM (property-difference-based averaging)

UEM Methodology:
---------------
The UEM provides an alternative extrapolation method from binary to multicomponent
systems. It uses the SAME Redlich-Kister polynomials for binary interactions, but
employs a different extrapolation scheme based on:

1. Property differences (δ) between components (calculated from binary parameters)
2. Contribution coefficients (r) that weight how third components affect binary pairs
3. Effective mole fractions that account for component similarity

This allows UEM to cover traditional extrapolation models as special cases while
providing better predictions for highly asymmetric systems.

Correct Terminology:
-------------------
- Binary description: Redlich-Kister polynomials (L0, L1, L2, ...)
- Multicomponent extrapolation: UEM method
- Complete name: Redlich-Kister-UEM

Key References:
--------------
- Chou, K. C. (2020). On the definition of the components' difference in properties
  in the unified extrapolation model. Fluid Phase Equilibria, 507, 112416.
- Chou, K. C., Wei, S. K. (2020). New expression for property difference in components
  for the Unified Extrapolation Model. Journal of Molecular Liquids, 298, 111951.
- Chou, K. C. et al. (2024). Thermochimica Acta, 179824.
"""
from sympy import Symbol, Add, Mul, Pow, Abs, exp, simplify, S, Piecewise, nan, Basic, Float
from pycalphad import variables as v
from pycalphad.core.utils import wrap_symbol
from pycalphad.variables import R
from tinydb import where
import logging

# Configure logging
logger = logging.getLogger(__name__)


def is_binary_in_phase(arr, comp1, comp2):
	"""
	Check if constituent array contains only the specified two components.

	This helper function is used to identify binary interaction parameters
	in the thermodynamic database that correspond to a specific pair of components.

	Parameters
	----------
	arr : list of lists
		Constituent array from database parameter (phase composition array)
	comp1 : str
		First component name
	comp2 : str
		Second component name

	Returns
	-------
	bool
		True if the array contains exactly the two specified components

	Examples
	--------
	>>> is_binary_in_phase([['AL', 'NI']], 'AL', 'NI')
	True
	>>> is_binary_in_phase([['AL', 'NI', 'CR']], 'AL', 'NI')
	False
	"""
	comps = set(str(s) for subl in arr for s in subl)
	return comp1 in comps and comp2 in comps and len(comps) == 2


_param_cache = {}

def wrap_parameter_safely(parameter):
	
	return parameter


def uem1_delta_expr(dbe, comp1, comp2, phase_name, T):
	"""
	Calculate the property difference expression between two components in UEM.

	The property difference (delta) is a key quantity in UEM that characterizes
	the dissimilarity between two components. It is calculated from the asymmetry
	of the binary excess Gibbs energy at the composition boundaries.

	Mathematical formulation:
	For a binary system i-j with excess Gibbs energy G_ex(x), the property
	difference is defined as:

		delta_ij = |dG_ex/dx|_{x=0} - |dG_ex/dx|_{x=1}| / (R*T)

	where the derivatives represent the limiting slopes at pure component limits.

	Parameters
	----------
	dbe : Database
		Thermodynamic database containing binary interaction parameters
	comp1 : str
		First component name
	comp2 : str
		Second component name
	phase_name : str
		Phase name for which to calculate property difference
	T : StateVariable
		Temperature variable (symbolic)

	Returns
	-------
	SymPy expression
		Normalized dimensionless property difference delta_ij
		Returns S.Zero if no binary parameters are found

	Notes
	-----
	The property difference quantifies how different two components are in terms
	of their interactions in the binary subsystem. A small delta indicates similar
	behavior (symmetric system), while a large delta indicates dissimilar behavior
	(asymmetric system).
	"""

	x = Symbol('x')
	G_ex = S.Zero

	# Query database for binary interaction parameters
	param_query = (
			(where('phase_name') == phase_name) &
			(where('parameter_type') == 'G') &
			(where('constituent_array').test(lambda arr: is_binary_in_phase(arr, comp1, comp2)))
	)
	params = dbe.search(param_query)

	if not params:
		logger.debug(f"No interaction parameters found for {comp1}-{comp2} in {phase_name}")
		return S.Zero

	# Construct excess Gibbs energy expression using Redlich-Kister formulation
	# G_ex = x*(1-x) * sum_n L_n * (2*x - 1)^n
	for p in params:
		try:
			order = p['parameter_order']  # Redlich-Kister order (0, 1, 2, ...)
			param = wrap_parameter_safely(p['parameter'])
			G_ex += Mul(x, 1 - x, param, Pow(2 * x - 1, order))
		except Exception as e:
			logger.warning(f"Error processing parameter for {comp1}-{comp2}: {str(e)}")
			continue

	# Calculate derivatives at composition boundaries
	# At x=0 (pure comp2): dG_ex/dx|_{x=0}
	dGdx_at_0 = G_ex.diff(x).subs(x, 0)

	# At x=1 (pure comp1): dG_ex/dx|_{x=1}
	# Using substitution x -> (1-x) for symmetry
	dGdx_at_1 = G_ex.subs(x, 1 - x).diff(x).subs(x, 0)

	# Calculate absolute difference and normalize by R*T
	delta = Abs(dGdx_at_0 - dGdx_at_1) / (R * T)
	normalized_delta = simplify(delta)

	return normalized_delta


def uem1_contribution_ratio(dbe, k, i, j, phase_name, T):
	"""
	Calculate the contribution coefficient of component k to the i-j binary pair in UEM.

	The contribution coefficient determines how much component k contributes to the
	effective mole fraction of component i when considering the i-j binary subsystem
	in a multicomponent mixture.

	Mathematical formulation:
		r_ki = (delta_kj / (delta_ki + delta_kj)) * exp(-delta_ki)

	where delta_ki and delta_kj are property differences between components.

	Physical interpretation:
	- When k is similar to i (small delta_ki), r_ki is larger (k contributes more to i)
	- When k is similar to j (small delta_kj), r_ki is smaller (k contributes less to i)
	- The exponential term exp(-delta_ki) enhances contribution when k and i are similar

	Parameters
	----------
	dbe : Database
		Thermodynamic database containing binary interaction parameters
	k : str
		Third component name (contributing component)
	i : str
		First component name in binary pair
	j : str
		Second component name in binary pair
	phase_name : str
		Phase name
	T : StateVariable
		Temperature variable (symbolic)

	Returns
	-------
	SymPy expression
		Contribution coefficient r_ki (dimensionless, typically between 0 and 1)

	Notes
	-----
	If both delta_ki and delta_kj are zero (all three components are identical),
	the function returns 0.5 (equal contribution).
	"""
	delta_ki = uem1_delta_expr(dbe, k, i, phase_name, T)
	delta_kj = uem1_delta_expr(dbe, k, j, phase_name, T)

	# Handle case where all components are identical
	if delta_ki == S.Zero and delta_kj == S.Zero:
		logger.debug(f"All deltas zero for {k}-{i}-{j}, using default ratio 0.5")
		return S.Half

	# Calculate contribution ratio with numerical stability check
	try:
		# r_ki = (delta_kj / (delta_ki + delta_kj)) * exp(-delta_ki)
		result = simplify((delta_kj / (delta_ki + delta_kj)) * exp(-delta_ki))
		logger.debug(f"Contribution ratio r_{k}{i} in {i}-{j} pair calculated")
		return result
	except Exception as e:
		logger.warning(f"Error calculating contribution ratio for {k}-{i}-{j}: {str(e)}")
		return S.Half


def construct_binary_excess(dbe, comp_i, comp_j, phase_name, x_eff_i, x_eff_j):
	"""
	Construct binary excess Gibbs energy expression using effective mole fractions.

	This function builds the excess Gibbs energy for a binary subsystem i-j using
	the Redlich-Kister polynomial formulation with effective mole fractions instead
	of actual mole fractions.

	Mathematical formulation:
		G_ex_ij = x_eff_i * x_eff_j * sum_n [L_n * (x_eff_i - x_eff_j)^n]

	where:
	- x_eff_i, x_eff_j are effective mole fractions that account for contributions
	  from other components in the multicomponent system
	- L_n are Redlich-Kister interaction parameters from the database

	Parameters
	----------
	dbe : Database
		Thermodynamic database containing binary interaction parameters
	comp_i : str
		First component name
	comp_j : str
		Second component name
	phase_name : str
		Phase name
	x_eff_i : SymPy expression
		Effective mole fraction of component i
	x_eff_j : SymPy expression
		Effective mole fraction of component j

	Returns
	-------
	SymPy expression
		Binary excess Gibbs energy expression (J/mol)

	Notes
	-----
	The use of effective mole fractions (rather than actual mole fractions) is the
	key innovation of UEM that allows proper extrapolation from binary to multicomponent
	systems.
	"""
	G_ex_ij = S.Zero

	# Query database for binary interaction parameters
	param_query = (
			(where('phase_name') == phase_name) &
			(where('parameter_type') == 'G') &
			(where('constituent_array').test(lambda arr: is_binary_in_phase(arr, comp_i, comp_j)))
	)
	params = dbe.search(param_query)

	if not params:
		logger.debug(f"No binary parameters found for {comp_i}-{comp_j} in {phase_name}")
		return S.Zero

	# Build Redlich-Kister sum with effective mole fractions
	for p in params:
		try:
			order = p['parameter_order']  # L0, L1, L2, ...
			param = wrap_parameter_safely(p['parameter'])

			# Add term: x_eff_i * x_eff_j * L_n * (x_eff_i - x_eff_j)^n
			term = Mul(x_eff_i, x_eff_j, param, Pow(x_eff_i - x_eff_j, order))
			G_ex_ij = Add(G_ex_ij, term)
		except Exception as e:
			logger.warning(f"Error processing parameter for {comp_i}-{comp_j}: {str(e)}")
			continue

	return G_ex_ij


def is_stable_expression(expr):
	"""
	Check if a symbolic expression is numerically stable.

	Parameters
	----------
	expr : SymPy expression
		Expression to check for numerical stability

	Returns
	-------
	bool
		True if expression appears numerically stable, False otherwise

	Notes
	-----
	Checks for common sources of numerical instability:
	- Complex infinities (zoo)
	- Infinities (oo, -oo)
	- NaN values
	- Negative powers (potential division by small numbers)
	"""
	from sympy import zoo, oo, nan, preorder_traversal

	# Check for infinities and NaN
	if expr.has(zoo) or expr.has(oo) or expr.has(nan):
		return False

	# Check for potentially problematic operations
	for node in preorder_traversal(expr):
		if node.is_Pow and node.args[1] < 0:  # Negative power indicates division
			return False
		if node.is_Mul:
			for arg in node.args:
				if arg.is_Pow and arg.args[1] < 0:
					return False

	return True


def get_uem1_excess_gibbs_expr(dbe, comps, phase_name, T):
	"""
	Construct UEM excess Gibbs energy expression for multicomponent solution phase.

	This is the main function that implements the Unified Extrapolation Model (UEM)
	to calculate excess Gibbs energy of multicomponent systems from binary subsystem
	parameters.

	Algorithm:
	1. For each binary pair (i,j) in the multicomponent system:
	   a. Calculate effective mole fractions by adding contributions from all
	      other components k based on their similarity to i and j
	   b. Normalize effective mole fractions to sum to 1 within the i-j subsystem
	   c. Construct binary excess energy using these effective mole fractions
	   d. Weight the binary contribution by a scaling factor
	2. Sum contributions from all binary pairs

	Mathematical formulation:
		G_ex_total = sum_over_pairs [(x_i * x_j) / (Xi_ij * Xj_ij)] * G_ex_ij(Xi_ij, Xj_ij)

	where:
	- x_i, x_j are actual mole fractions
	- x_eff_i = x_i + sum_k(r_ki * x_k) for all k != i,j
	- Xi_ij = x_eff_i / (x_eff_i + x_eff_j) (normalized effective mole fraction)
	- r_ki is the contribution coefficient of k to i in the i-j pair
	- G_ex_ij is the binary excess energy from database parameters

	Parameters
	----------
	dbe : Database
		Thermodynamic database containing binary interaction parameters
	comps : list of str
		List of component names (excluding vacancies)
	phase_name : str
		Name of the solution phase
	T : StateVariable
		Temperature variable (symbolic)

	Returns
	-------
	SymPy expression
		Total excess Gibbs energy expression for the multicomponent system (J/mol)

	Examples
	--------
	>>> from pycalphad import Database, variables as v
	>>> dbf = Database('example.tdb')
	>>> comps = ['AL', 'CR', 'NI']
	>>> expr = get_uem1_excess_gibbs_expr(dbf, comps, 'LIQUID', v.T)

	Notes
	-----
	- For binary systems, UEM reduces to the standard Redlich-Kister formulation
	- The model requires binary interaction parameters for all component pairs
	- Missing binary parameters are treated as zero (ideal mixing)
	- Numerical stability is ensured by handling division by zero cases
	"""

	# Create mole fraction symbols for all components
	x = {comp: v.X(comp) for comp in comps}
	expr_list = []

	# Iterate over all unique binary pairs
	for i_idx in range(len(comps)):
		for j_idx in range(i_idx + 1, len(comps)):
			comp_i = comps[i_idx]
			comp_j = comps[j_idx]

			logger.debug(f"Processing binary pair {comp_i}-{comp_j} in {phase_name}")

			# Start with actual mole fractions
			x_eff_i = x[comp_i]
			x_eff_j = x[comp_j]

			# Add contributions from all other components
			for k in comps:
				if k not in [comp_i, comp_j]:
					try:
						# Calculate how much k contributes to i and j
						r_ki = uem1_contribution_ratio(dbe, k, comp_i, comp_j, phase_name, T)
						r_kj = uem1_contribution_ratio(dbe, k, comp_j, comp_i, phase_name, T)

						# Add weighted contributions to effective mole fractions
						x_eff_i += r_ki * x[k]
						x_eff_j += r_kj * x[k]

					except Exception as e:
						logger.warning(f"Error calculating contribution ratios for {k} in {comp_i}-{comp_j}: {str(e)}")
						continue

			# Normalize effective mole fractions to the i-j subsystem
			# Xi_ij = x_eff_i / (x_eff_i + x_eff_j)
			denominator = x_eff_i + x_eff_j

			Xi_ij = x_eff_i / denominator
			Xj_ij = x_eff_j / denominator

			# Construct binary excess Gibbs energy with normalized effective mole fractions
			G_ex_ij = construct_binary_excess(dbe, comp_i, comp_j, phase_name, Xi_ij, Xj_ij)

			# Clean up any NaN values
			G_ex_ij = simplify(G_ex_ij.subs({S.NaN: S.Zero}))

			# Calculate scaling factor: (x_i * x_j) / (Xi_ij * Xj_ij)
			# This ensures proper weighting of binary contributions
			if Xi_ij == S.Zero or Xj_ij == S.Zero:
				ratio = S.Zero
			else:
				ratio = (x[comp_i] * x[comp_j]) / (Xi_ij * Xj_ij)

			# Add weighted binary contribution to total expression
			expr_list.append(G_ex_ij * ratio)

	# Sum all binary pair contributions
	total_expr = Add(*expr_list)

	# Final cleanup: replace any remaining NaN with zero
	total_expr = total_expr.subs(nan, 0)

	logger.info(f"UEM excess Gibbs energy expression constructed for {len(comps)}-component system in {phase_name}")

	return total_expr
