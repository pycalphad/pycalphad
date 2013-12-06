/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition for CompositionSet class

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libtdb/include/logging.hpp"
#include "libgibbs/include/compositionset.hpp"
#include "libgibbs/include/utils/math_expr.hpp"
#include <boost/algorithm/string/predicate.hpp>

using boost::multi_index_container;
using namespace boost::multi_index;

CompositionSet::CompositionSet(
		const Phase &phaseobj,
		const parameter_set &pset,
		const sublattice_set &sublset,
		boost::bimap<std::string, int> const &main_indices) {
	BOOST_LOG_NAMED_SCOPE("CompositionSet::CompositionSet");
	logger comp_log(journal::keywords::channel = "optimizer");
	cset_name = phaseobj.name();

	// Now initialize the appropriate models
	models["PURE_ENERGY"] = std::unique_ptr<EnergyModel>(new PureCompoundEnergyModel(phaseobj.name(), sublset, pset));
	models["IDEAL_MIX"] = std::unique_ptr<EnergyModel>(new IdealMixingModel(phaseobj.name(), sublset));
	models["REDLICH_KISTER"] = std::unique_ptr<EnergyModel>(new RedlichKisterExcessEnergyModel(phaseobj.name(), sublset, pset));
	models["IHJ_MAGNETIC"] = std::unique_ptr<EnergyModel>(new IHJMagneticModel(phaseobj.name(), sublset, pset,
					phaseobj.magnetic_afm_factor, phaseobj.magnetic_sro_enthalpy_order_fraction));

	for (auto i = models.begin(); i != models.end(); ++i) {
		auto symbol_table = i->second->get_symbol_table();
		symbols.insert(symbol_table.begin(), symbol_table.end()); // copy model symbols into main symbol table
		// TODO: we don't check for duplicate symbols at all here...models police themselves to avoid collisions
		// One idea: put all symbols into model-specific namespaces
		for (auto j = symbol_table.begin(); j != symbol_table.end(); ++j) {
			BOOST_LOG_SEV(comp_log, debug) << "added symbol " << j->first << " to composition set " << cset_name << ": "<< j->second.get();
		}
	}

	// Calculate first derivative ASTs of all variables
	for (auto i = main_indices.left.begin(); i != main_indices.left.end(); ++i) {
		std::list<std::string> diffvars;
		diffvars.push_back(i->first);
		if (!boost::algorithm::starts_with(i->first, cset_name)) {
			// the differentiating variable doesn't come from this composition set
			// the derivative should be zero, so skip calculation
			continue;
		}
		for (auto j = models.cbegin(); j != models.cend(); ++j) {
			boost::spirit::utree difftree;
			if (i->first == (cset_name + "_FRAC")) {
				// the derivative w.r.t the phase fraction is just the energy of this phase
				difftree = j->second->get_ast();
			}
			else {
				difftree = simplify_utree(differentiate_utree(j->second->get_ast(), i->first, symbols));
			}
			if (!is_zero_tree(difftree)) {
				tree_data.insert(ast_entry(diffvars, j->first, difftree));
			}

			// Calculate second derivative ASTs of all variables (doesn't include constraint contribution)
			for (auto k = main_indices.left.begin(); k != main_indices.left.end(); ++k) {
				// second derivative of obj function w.r.t i,j
				if (i->second > k->second) continue; // skip upper triangular
				std::list<std::string> second_diffvars = diffvars;
				second_diffvars.push_back(k->first);
				if (k->first == (cset_name + "_FRAC")) {
					// second derivative w.r.t phase fraction is zero
				}
				else if (!boost::algorithm::starts_with(k->first, cset_name)) {
					// the differentiating variable doesn't come from this composition set
					// the derivative should be zero, so skip calculation
				}
				else {
					boost::spirit::utree second_difftree = simplify_utree(differentiate_utree(difftree, k->first, symbols));
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
		boost::bimap<std::string, int> const &main_indices,
		double* const x) const {
	BOOST_LOG_NAMED_SCOPE("CompositionSet::evaluate_objective(evalconditions const& conditions,boost::bimap<std::string, int> const &main_indices,double* const x)");
	logger comp_log;
	BOOST_LOG_SEV(comp_log, debug) << "enter";
	double objective = 0;
	const std::string compset_name(cset_name + "_FRAC");
	BOOST_LOG_SEV(comp_log, debug) << "compset_name: " << compset_name;

	for (auto i = models.cbegin(); i != models.cend(); ++i) {
		// multiply by phase fraction
		objective += x[main_indices.left.at(compset_name)] *
				process_utree(i->second->get_ast(), conditions, main_indices, symbols, x).get<double>();
	}

	BOOST_LOG_SEV(comp_log, debug) << "returning";
	return objective;
}
double CompositionSet::evaluate_objective(
		evalconditions const &conditions, std::map<std::string,double> const &variables) const {
	// Need to translate this variable map into something process_utree can understand
	BOOST_LOG_NAMED_SCOPE("CompositionSet::evaluate_objective(evalconditions const &conditions, std::map<std::string,double> const &variables)");
	logger comp_log(journal::keywords::channel = "optimizer");
	BOOST_LOG_SEV(comp_log, debug) << "enter";
	double vars[variables.size()]; // Create Ipopt-style double array
	boost::bimap<std::string, int> main_indices;
	typedef boost::bimap<std::string, int>::value_type position;
	for (auto i = variables.begin(); i != variables.end(); ++i) {
		vars[std::distance(variables.begin(),i)] = i->second; // Copy values into array
		BOOST_LOG_SEV(comp_log, debug) << "main_indices.insert(" << i->first << ", " << std::distance(variables.begin(), i) << ")";
		main_indices.insert(position(i->first, std::distance(variables.begin(),i))); // Create fictitious indices
	}
	for (auto i = main_indices.left.begin(); i != main_indices.left.end(); ++i) {
		BOOST_LOG_SEV(comp_log, debug) << i->first << " -> " << i->second;
	}
	BOOST_LOG_SEV(comp_log, debug) << "returning";
	return evaluate_objective(conditions, main_indices, vars);
}

std::map<int,double> CompositionSet::evaluate_objective_gradient(
		evalconditions const& conditions, boost::bimap<std::string, int> const &main_indices, double* const x) const {
	std::map<int,double> retmap;
	boost::multi_index::index<ast_set,ast_deriv_order_index>::type::const_iterator ast_begin,ast_end;
	ast_begin = get<ast_deriv_order_index>(tree_data).lower_bound(1);
	ast_end = get<ast_deriv_order_index>(tree_data).upper_bound(1);
	const std::string compset_name(cset_name + "_FRAC");

	for (auto i = main_indices.left.begin(); i != main_indices.left.end(); ++i) {
		retmap[i->second] = 0; // initialize all indices as zero
	}
	for (ast_set::const_iterator i = ast_begin; i != ast_end; ++i) {
		const double diffvalue = process_utree(i->ast, conditions, main_indices, symbols, x).get<double>();
		const std::string diffvar = *(i->diffvars.cbegin()); // get differentiating variable
		const int varindex = main_indices.left.at(diffvar);
		if (diffvar != compset_name) {
			retmap[varindex] += x[main_indices.left.at(compset_name)] * diffvalue; // multiply derivative by phase fraction
		}
		else {
			// don't multiply derivative by phase fraction because this is the derivative w.r.t phase fraction
			retmap[varindex] += diffvalue;
		}
	}

	return retmap;
}

std::map<std::list<int>,double> CompositionSet::evaluate_objective_hessian(
			evalconditions const& conditions,
			boost::bimap<std::string, int> const &main_indices,
			double* const x) const {
	std::map<std::list<int>,double> retmap;
	boost::multi_index::index<ast_set,ast_deriv_order_index>::type::const_iterator ast_begin,ast_end;
	ast_begin = get<ast_deriv_order_index>(tree_data).lower_bound(2);
	ast_end = get<ast_deriv_order_index>(tree_data).upper_bound(2);
	const std::string compset_name(cset_name + "_FRAC");

	for (auto i = main_indices.left.begin(); i != main_indices.left.end(); ++i) {
		for (auto j = main_indices.left.begin(); j != main_indices.left.end(); ++j) {
			if (i->second > j->second) continue; // skip upper triangular
			const std::list<int> searchlist {i->second,j->second};
			retmap[searchlist] = 0; // initialize all indices as zero
		}
	}

	for (ast_set::const_iterator i = ast_begin; i != ast_end; ++i) {
		const double diffvalue = process_utree(i->ast, conditions, main_indices, symbols, x).get<double>();
		const std::string diffvar1 = *(i->diffvars.cbegin());
		const std::string diffvar2 = *(++(i->diffvars.cbegin()));
		const int varindex1 = main_indices.left.at(diffvar1);
		const int varindex2 = main_indices.left.at(diffvar2);
		std::list<int> searchlist;
		if (varindex1 <= varindex2) searchlist = {varindex1,varindex2};
		else searchlist = {varindex2, varindex1};
		// multiply derivative by phase fraction
		if (diffvar1 == compset_name || diffvar2 == compset_name) {
			retmap[searchlist] += diffvalue;
		}
		else retmap[searchlist] += x[main_indices.left.at(compset_name)] * diffvalue;
	}
	return retmap;
}

std::set<std::list<int>> CompositionSet::hessian_sparsity_structure(
		boost::bimap<std::string, int> const &main_indices) const {
	std::set<std::list<int>> retset;
	boost::multi_index::index<ast_set,ast_deriv_order_index>::type::const_iterator ast_begin,ast_end;
	ast_begin = get<ast_deriv_order_index>(tree_data).lower_bound(2);
	ast_end = get<ast_deriv_order_index>(tree_data).upper_bound(2);
	for (ast_set::const_iterator i = ast_begin; i != ast_end; ++i) {
		const std::string diffvar1 = *(i->diffvars.cbegin());
		const std::string diffvar2 = *(++(i->diffvars.cbegin()));
		const int varindex1 = main_indices.left.at(diffvar1);
		const int varindex2 = main_indices.left.at(diffvar2);
		std::list<int> nonzero_entry;
		if (varindex1 <= varindex2) nonzero_entry = {varindex1,varindex2};
		else nonzero_entry = {varindex2, varindex1};
		retset.insert(nonzero_entry);
	}
	return retset;
}
