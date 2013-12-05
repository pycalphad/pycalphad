/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

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
#include <memory>
#include <set>
#include <list>
#include <boost/bimap.hpp>

// A CompositionSet works with libtdb's Phase class
// Its purpose is to handle the optimizer's specific configuration for the given conditions and models
// Multiple CompositionSets of the same Phase can be created to handle miscibility gaps
class CompositionSet {
public:
	double evaluate_objective(evalconditions const&, boost::bimap<std::string, int> const &, double* const) const;
	double evaluate_objective(evalconditions const &, std::map<std::string,double> const &) const;
	std::map<int,double> evaluate_objective_gradient(
			evalconditions const&, boost::bimap<std::string, int> const &, double* const) const;
	std::map<std::list<int>,double> evaluate_objective_hessian(
			evalconditions const&, boost::bimap<std::string, int> const &, double* const) const;
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
	}
	CompositionSet& operator=(CompositionSet &&other) {
		cset_name = std::move(other.cset_name);
		models = std::move(other.models);
		jac_g_trees = std::move(other.jac_g_trees);
		hessian_data = std::move(other.hessian_data);
		tree_data = std::move(other.tree_data);
		first_derivatives = std::move(other.first_derivatives);
	}
	CompositionSet(const CompositionSet &) = delete;
	CompositionSet& operator=(const CompositionSet &) = delete;
private:
	std::string cset_name;
	std::map<std::string,std::unique_ptr<EnergyModel>> models;
	std::map<int,boost::spirit::utree> first_derivatives;
	std::vector<jacobian_entry> jac_g_trees;
	hessian_set hessian_data;
	ast_set tree_data;
	ASTSymbolMap symbols; // maps special symbols to ASTs and their derivatives
};

#endif
