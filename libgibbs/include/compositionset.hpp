/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// declaration for CompositionSet class

#ifndef COMPOSITIONSET_INCLUDED
#define COMPOSITIONSET_INCLUDED

#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/ast_set.hpp"
#include "libtdb/include/structure.hpp"
#include <memory>

// A CompositionSet works with libtdb's Phase class
// Its purpose is to handle the optimizer's specific configuration for the given conditions and models
// Multiple CompositionSets of the same Phase can be created to handle miscibility gaps
class CompositionSet {
public:
	double evaluate_objective(
			evalconditions const&, std::map<std::string, int> const &, double* const) const;
	std::map<int,double> evaluate_objective_gradient(
			evalconditions const&, std::map<std::string, int> const &, double* const) const;
	std::vector<double> evaluate_constraints();
	std::vector<double> evaluate_jacobian_of_constraints();
	std::vector<double> evaluate_hessian();
	std::string name() { return cset_name; }
	// make CompositionSet from existing Phase
	CompositionSet(
			const Phase &phaseobj,
			const parameter_set &pset,
			const sublattice_set &sublset,
			std::map<std::string, int> const &main_indices);
private:
	std::string cset_name;
	std::map<std::string,std::unique_ptr<EnergyModel>> models;
	std::map<int,boost::spirit::utree> first_derivatives;
	std::vector<jacobian_entry> jac_g_trees;
	hessian_set hessian_data;
	ast_set tree_data;
	// Block copy ctor
	CompositionSet(const CompositionSet&);
	// Block assignment operator
	CompositionSet& operator=(const CompositionSet&);
};

#endif
