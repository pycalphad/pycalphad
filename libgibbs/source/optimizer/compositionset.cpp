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
		std::vector<std::string> diffvars;
		diffvars.push_back(i->first);
		if (!boost::algorithm::starts_with(i->first, cset_name)) {
			// the differentiating variable doesn't come from this composition set
			// the derivative should be zero, so skip calculation
			tree_data.insert(ast_entry(1, diffvars, std::string(), boost::spirit::utree(0)));
			continue;
		}
		for (auto j = models.cbegin(); j != models.cend(); ++j) {
			boost::spirit::utree difftree = simplify_utree(differentiate_utree(j->second->get_ast(), i->first));
			if (is_zero_tree(difftree)) continue;
			tree_data.insert(ast_entry(1, diffvars, j->first, difftree));
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

std::vector<double> CompositionSet::evaluate_objective_gradient(
			evalconditions const& conditions, std::map<std::string, int> const &main_indices, double* const x) const {
	std::vector<double> retvec;
	return retvec;
}
