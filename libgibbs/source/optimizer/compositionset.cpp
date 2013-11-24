/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition for CompositionSet class

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/compositionset.hpp"
#include "libtdb/include/utils/math_expr.hpp"
#include <boost/algorithm/string/predicate.hpp>

using boost::multi_index_container;
using namespace boost::multi_index;

CompositionSet::CompositionSet(
		const Phase &phaseobj,
		const parameter_set &pset,
		const sublattice_set &sublset,
		std::map<std::string, int> const &main_indices) {
	cset_name = phaseobj.name();

	// Now initialize the appropriate models
	models["PURE_ENERGY"] = std::unique_ptr<EnergyModel>(new PureCompoundEnergyModel(phaseobj.name(), sublset, pset));
	models["IDEAL_MIX"] = std::unique_ptr<EnergyModel>(new IdealMixingModel(phaseobj.name(), sublset));
	models["REDLICH_KISTER"] = std::unique_ptr<EnergyModel>(new RedlichKisterExcessEnergyModel(phaseobj.name(), sublset, pset));
	models["IHJ_MAGNETIC"] = std::unique_ptr<EnergyModel>(new IHJMagneticModel(phaseobj.name(), sublset, pset,
					phaseobj.magnetic_afm_factor, phaseobj.magnetic_sro_enthalpy_order_fraction));

	// Calculate first derivative ASTs of all variables
	for (auto i = main_indices.cbegin(); i != main_indices.cend(); ++i) {
		std::list<std::string> diffvars;
		diffvars.push_back(i->first);
		if (!boost::algorithm::starts_with(i->first, cset_name)) {
			// the differentiating variable doesn't come from this composition set
			// the derivative should be zero, so skip calculation
			tree_data.insert(ast_entry(diffvars, std::string(), boost::spirit::utree(0)));
			continue;
		}
		for (auto j = models.cbegin(); j != models.cend(); ++j) {
			boost::spirit::utree difftree;
			if (i->first == (cset_name + "_FRAC")) {
				// the derivative w.r.t the phase fraction is just the energy of this phase
				difftree = j->second->get_ast();
			}
			else {
				difftree = simplify_utree(differentiate_utree(j->second->get_ast(), i->first));
			}
			if (!is_zero_tree(difftree)) {
				tree_data.insert(ast_entry(diffvars, j->first, difftree));
			}

			// Calculate second derivative ASTs of all variables (doesn't include constraint contribution)
			for (auto k = main_indices.cbegin(); k != main_indices.cend(); ++k) {
				// second derivative of obj function w.r.t i,j
				if (i->second > k->second) continue; // skip upper triangular
				std::list<std::string> second_diffvars = diffvars;
				second_diffvars.push_back(k->first);
				if (k->first == (cset_name + "_FRAC")) {
					// second derivative w.r.t phase fraction is zero
					tree_data.insert(ast_entry(second_diffvars, j->first, boost::spirit::utree(0)));
				}
				else if (!boost::algorithm::starts_with(k->first, cset_name)) {
					// the differentiating variable doesn't come from this composition set
					// the derivative should be zero, so skip calculation
					tree_data.insert(ast_entry(second_diffvars, j->first, boost::spirit::utree(0)));
				}
				else {
					boost::spirit::utree second_difftree = simplify_utree(differentiate_utree(difftree, k->first));
					if (!is_zero_tree(second_difftree)) {
						tree_data.insert(ast_entry(second_diffvars, j->first, second_difftree));
					}
				}
			}
		}
	}
}

double CompositionSet::evaluate_objective(
		evalconditions const& conditions,
		std::map<std::string, int> const &main_indices,
		double* const x) const {
	double objective = 0;
	const std::string compset_name(cset_name + "_FRAC");

	for (auto i = models.cbegin(); i != models.cend(); ++i) {
		// multiply by phase fraction
		objective += x[(*main_indices.find(compset_name)).second] *
				process_utree(i->second->get_ast(), conditions, main_indices, x).get<double>();
	}

	return objective;
}

std::map<int,double> CompositionSet::evaluate_objective_gradient(
		evalconditions const& conditions, std::map<std::string, int> const &main_indices, double* const x) const {
	std::map<int,double> retmap;
	boost::multi_index::index<ast_set,ast_deriv_order_index>::type::const_iterator ast_begin,ast_end;
	ast_begin = get<ast_deriv_order_index>(tree_data).lower_bound(1);
	ast_end = get<ast_deriv_order_index>(tree_data).upper_bound(1);
	const std::string compset_name(cset_name + "_FRAC");

	for (auto i = main_indices.cbegin(); i != main_indices.cend(); ++i) {
		retmap[i->second] = 0; // initialize all indices as zero
	}
	for (auto i = ast_begin; i != ast_end; ++i) {
		const double diffvalue = process_utree(i->ast, conditions, main_indices, x).get<double>();
		const std::string diffvar = *(i->diffvars.cbegin()); // get differentiating variable
		const int varindex = (*main_indices.find(diffvar)).second;
		if (diffvar != compset_name) {
			retmap[varindex] += x[(*main_indices.find(compset_name)).second] * diffvalue; // multiply derivative by phase fraction
		}
		else {
			// don't multiply derivative by phase fraction because this is the derivative w.r.t phase fraction
			retmap[varindex] += diffvalue;
		}
	}

	return retmap;
}

std::map<std::list<int,int>,double> CompositionSet::evaluate_objective_hessian(
			evalconditions const& conditions,
			std::map<std::string, int> const &main_indices,
			double* const x) const {
	std::map<std::list<int,int>,double> retmap;
	boost::multi_index::index<ast_set,ast_deriv_order_index>::type::const_iterator ast_begin,ast_end;
	ast_begin = get<ast_deriv_order_index>(tree_data).lower_bound(2);
	ast_end = get<ast_deriv_order_index>(tree_data).upper_bound(2);
	const std::string compset_name(cset_name + "_FRAC");

	for (auto i = main_indices.cbegin(); i != main_indices.cend(); ++i) {
		for (auto j = main_indices.cbegin(); j != main_indices.cend(); ++j) {
			if (i->second > j->second) continue; // skip upper triangular
			retmap[std::list<int,int>({i->second,j->second})] = 0; // initialize all indices as zero
		}
	}

	for (auto i = ast_begin; i != ast_end; ++i) {
		const double diffvalue = process_utree(i->ast, conditions, main_indices, x).get<double>();
		const std::string diffvar1 = *(i->diffvars.cbegin());
		const std::string diffvar2 = *(i->diffvars.cbegin()+1);
		const int varindex1 = (*main_indices.find(diffvar1)).second;
		const int varindex2 = (*main_indices.find(diffvar2)).second;
		// multiply derivative by phase fraction
		retmap[std::list<int,int>({varindex1,varindex2})] += x[(*main_indices.find(compset_name)).second] * diffvalue;
	}
	return retmap;
}
