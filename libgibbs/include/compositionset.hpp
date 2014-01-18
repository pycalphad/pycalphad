/*=============================================================================
	Copyright (c) 2012-2014 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// declaration for CompositionSet class

#ifndef COMPOSITIONSET_INCLUDED
#define COMPOSITIONSET_INCLUDED

#include "libgibbs/include/models.hpp"
#include "libgibbs/include/constraint.hpp"
#include "libgibbs/include/optimizer/ast_set.hpp"
#include "libgibbs/include/conditions.hpp"
#include "libgibbs/include/utils/ast_caching.hpp"
#include "libtdb/include/structure.hpp"
#include <boost/bimap.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <memory>
#include <set>
#include <list>
#include <vector>
#include <utility>

// A CompositionSet works with libtdb's Phase class
// Its purpose is to handle the optimizer's specific configuration for the given conditions and models
// Multiple CompositionSets of the same Phase can be created to handle miscibility gaps
class CompositionSet {
public:
	double evaluate_objective(evalconditions const&, boost::bimap<std::string, int> const &, double* const) const;
	double evaluate_objective(evalconditions const &, std::map<std::string,double> const &) const;
	std::map<int,double> evaluate_objective_gradient(
			evalconditions const&, boost::bimap<std::string, int> const &, double* const) const;
	std::map<int,double> evaluate_objective_gradient(
			evalconditions const &, std::map<std::string,double> const &) const;
	std::vector<double> evaluate_internal_objective_gradient(
			evalconditions const& conditions, double* const) const;
	std::map<std::list<int>,double> evaluate_objective_hessian(
			evalconditions const&, boost::bimap<std::string, int> const &, double* const) const;
	boost::numeric::ublas::symmetric_matrix<double,boost::numeric::ublas::lower> evaluate_objective_hessian_matrix(
				evalconditions const& conditions,
				boost::bimap<std::string, int> const &main_indices,
				std::vector<double> const &x) const;
	std::set<std::list<int>> hessian_sparsity_structure(boost::bimap<std::string, int> const &) const;
	std::string name() const { return cset_name; }
	// make CompositionSet from existing Phase
	CompositionSet(
			const Phase &phaseobj,
			const parameter_set &pset,
			const sublattice_set &sublset,
			boost::bimap<std::string, int> const &main_indices);

	CompositionSet() { }

	CompositionSet(CompositionSet &&other) {
		cset_name = std::move(other.cset_name);
		models = std::move(other.models);
		jac_g_trees = std::move(other.jac_g_trees);
		hessian_data = std::move(other.hessian_data);
		tree_data = std::move(other.tree_data);
		first_derivatives = std::move(other.first_derivatives);
		symbols = std::move(other.symbols);
		cm = std::move(other.cm);
		phase_indices = std::move(other.phase_indices);
		constraint_null_space_matrix = std::move(other.constraint_null_space_matrix);
		constraint_particular_solution = std::move(other.constraint_particular_solution);
		constraint_extents = std::move(other.constraint_extents);
	}
	CompositionSet& operator=(CompositionSet &&other) {
		cset_name = std::move(other.cset_name);
		models = std::move(other.models);
		jac_g_trees = std::move(other.jac_g_trees);
		hessian_data = std::move(other.hessian_data);
		tree_data = std::move(other.tree_data);
		first_derivatives = std::move(other.first_derivatives);
		symbols = std::move(other.symbols);
		cm = std::move(other.cm);
		phase_indices = std::move(other.phase_indices);
		constraint_null_space_matrix = std::move(other.constraint_null_space_matrix);
		constraint_particular_solution = std::move(other.constraint_particular_solution);
		constraint_extents = std::move(other.constraint_extents);
	}
	const std::vector<jacobian_entry>& get_jacobian() const { return jac_g_trees; };
	const std::vector<Constraint>& get_constraints() const { return cm.constraints; };
	const boost::bimap<std::string, int>& get_variable_map() const { return phase_indices; };
	const boost::numeric::ublas::matrix<double>& get_constraint_null_space_matrix() const { return constraint_null_space_matrix; };
	const std::vector<std::pair<double,double> >& get_constraint_extents() const { return constraint_extents; };
	const ASTSymbolMap& get_symbols() const { return symbols; };
	CompositionSet(const CompositionSet &) = delete;
	CompositionSet& operator=(const CompositionSet &) = delete;
private:
	std::string cset_name;
	std::map<std::string,std::unique_ptr<EnergyModel>> models;
	std::map<int,boost::spirit::utree> first_derivatives;
	boost::bimap<std::string, int> phase_indices;
	std::vector<jacobian_entry> jac_g_trees;
	hessian_set hessian_data;
	ast_set tree_data;
	ASTSymbolMap symbols; // maps special symbols to ASTs and their derivatives
	ConstraintManager cm; // handles constraints internal to the phase, e.g., site fraction balances
	void build_constraint_basis_matrices(sublattice_set const &sublset);
	boost::numeric::ublas::matrix<double> constraint_null_space_matrix;
	boost::numeric::ublas::vector<double> constraint_particular_solution;
	std::vector<std::pair<double,double> > constraint_extents;
};

#endif
