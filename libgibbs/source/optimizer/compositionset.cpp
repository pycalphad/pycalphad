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
	for (auto i = main_indices.begin(); i != main_indices.end(); ++i) {
		std::list<std::string> diffvars;
		diffvars.push_back(i->first);
		if (!boost::algorithm::starts_with(i->first, cset_name)) {
			// the differentiating variable doesn't come from this composition set
			// the derivative should be zero, so skip calculation
			tree_data.insert(ast_entry(diffvars, std::string(), boost::spirit::utree(0)));
			continue;
		}
		for (auto j = models.cbegin(); j != models.cend(); ++j) {
			if (i->first == (cset_name + "_FRAC")) {
				// the derivative w.r.t the phase fraction is just the energy of this phase
				tree_data.insert(ast_entry(diffvars, j->first, j->second->get_ast()));
				continue;
			}
			boost::spirit::utree difftree = simplify_utree(differentiate_utree(j->second->get_ast(), i->first));
			if (is_zero_tree(difftree)) continue;
			tree_data.insert(ast_entry(diffvars, j->first, difftree));
		}
	}
}

double CompositionSet::evaluate_objective(
		evalconditions const& conditions,
		std::map<std::string, int> const &main_indices,
		double* const modelvars) const {
	double objective = 0;

	for (auto i = models.cbegin(); i != models.cend(); ++i) {
		objective += process_utree(i->second->get_ast(), conditions, main_indices, modelvars).get<double>();
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
