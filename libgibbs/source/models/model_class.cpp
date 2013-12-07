/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/models.hpp"
#include "libtdb/include/logging.hpp"
#include <string>
#include <map>
#include <algorithm>
#include <sstream>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/tuple/tuple.hpp>

using boost::spirit::utree;
typedef boost::spirit::utree_type utree_type;
using boost::multi_index_container;
using namespace boost::multi_index;


// count the total number of "mixing" sites in a sublattice set
// non-mixing sites are sublattices with only vacancies in them
double EnergyModel::count_mixing_sites(const sublattice_set_view &ssv) {
	int curindex = 0;
	int sitecount = 0;
	boost::multi_index::index<sublattice_set_view,myindex>::type::iterator ic0,ic1;
	ic0 = get<myindex>(ssv).lower_bound(curindex);
	ic1 = get<myindex>(ssv).upper_bound(curindex);

	// build subview to only "real" sublattices (exclude fake -1 index)
	while (ic0 != ic1) {
		int speccount = std::distance(ic0,ic1);
		if (!(speccount == 1 && (*ic0)->species == "VA")) {
			// only count non-pure vacancy sites
			sitecount += (*ic0)->num_sites;
		}
		++curindex;
		ic0 = get<myindex>(ssv).lower_bound(curindex);
		ic1 = get<myindex>(ssv).upper_bound(curindex);
	}
	return sitecount;
}

// helper function to add multiplicative factors of (y_i - y_j)**k
utree EnergyModel::add_interaction_factor(const std::string &lhs_varname, const std::string &rhs_varname, const double &degree, const utree &input_tree) {
	if (degree == 0) return input_tree;
	utree temp_tree, power_tree, ret_tree;
	temp_tree.push_back("-");
	temp_tree.push_back(lhs_varname);
	temp_tree.push_back(rhs_varname);
	power_tree.push_back("**");
	power_tree.push_back(temp_tree);
	power_tree.push_back(degree);
	if (degree == 1) power_tree = temp_tree;
	ret_tree.push_back("*");
	ret_tree.push_back(power_tree);
	ret_tree.push_back(input_tree);
	return ret_tree;
}

// Normalize by the total number of mixing sites
void EnergyModel::normalize_utree(utree &input_tree, const sublattice_set_view &ssv) {
	utree temp;
	temp.push_back("/");
	temp.push_back(input_tree);
	temp.push_back(count_mixing_sites(ssv));
	input_tree.swap(temp);
}

utree EnergyModel::find_parameter_ast(const sublattice_set_view &subl_view, const parameter_set_view &param_view) {
	BOOST_LOG_NAMED_SCOPE("EnergyModel::find_parameter_ast");
	logger model_log(journal::keywords::channel = "optimizer");
	BOOST_LOG_SEV(model_log, debug) << "enter";
	std::vector<const Parameter*> matches;
	journal::src::severity_channel_logger<severity_level> opt_log(journal::keywords::channel = "optimizer");
	int sublcount = 0;
	std::vector<std::vector<std::string>> search_config;
	auto subl_start = get<myindex>(subl_view).begin();
	auto subl_end = get<myindex>(subl_view).end();
	// TODO: parameter search would be better with a new index that was a derived key of the sublattice count
	// By the time we get here, we've already been filtered to the correct phase and parameter type
	auto param_iter = get<phase_index>(param_view).begin();
	auto param_end = get<phase_index>(param_view).end();

	// Build search configuration
	while (subl_start != subl_end) {
		int index = (*subl_start)->index;

		if (index < 0) {
			++subl_start;
			continue; // skip "fake" negative indices
		}

		// check if the current index exceeds the known sublattice count
		// expand the size of the vector of vectors by an amount equal to the difference
		// NOTE: if we don't do this, we will crash when we attempt to push_back()
		for (auto i = sublcount; i < (index+1); ++i) {
			std::vector<std::string> tempvec;
			search_config.push_back(tempvec);
			sublcount = index+1;
		}

		// Add species to the parameter search configuration
		search_config[(*subl_start)->index].push_back((*subl_start)->species);

		// Sort the sublattice with the newly added element
		//std::sort(search_config[(*subl_start)->index].begin(), search_config[(*subl_start)->index].end());


		++subl_start;
	}


	// Now that we have a search configuration, search through the parameters in param_view

	while (param_iter != param_end) {
		if (search_config.size() != (*param_iter)->constituent_array.size()) {
			// skip if sublattice counts do not match
			std::cout << "paramskip for sublattice mismatch: " << search_config.size() << " != " << (*param_iter)->constituent_array.size() << std::endl;
			std::cout << "search_config: ";
			for (auto i = search_config.begin(); i != search_config.end(); ++i) {
				for (auto j = (*i).begin(); j != (*i).end(); ++j) {
					std::cout << (*j) << ",";
				}
				std::cout << ":";
			}
			std::cout << std::endl;
			++param_iter;
			continue;
		}
		// We cannot do a direct comparison of the nested vectors because
		//    we must check the case where the wildcard '*' is used in some sublattices in the parameter
		bool isvalid = true;
		auto array_begin = (*param_iter)->constituent_array.begin();
		auto array_iter = array_begin;
		auto array_end = (*param_iter)->constituent_array.end();

		// Now perform a sublattice comparison
		while (array_iter != array_end) {
			const std::string firstelement = *(*array_iter).cbegin();
			// if the parameter sublattices don't match, or the parameter isn't using a wildcard, do not match
			// NOTE: A wildcard will not match if we are looking for a 2+ species interaction in that sublattice
			// The interaction must be explicitly stated to match
			if (!(
				(*array_iter) == search_config[std::distance(array_begin,array_iter)])
				|| (firstelement == "*" && search_config[std::distance(array_begin,array_iter)].size() == 1)
				) {
				isvalid = false;
				break;
			}
			++array_iter;
		}
		if (isvalid) {
			// We found a valid parameter, save it
			matches.push_back(*param_iter);
		}

		++param_iter;
	}

	if (matches.size() >= 1) {
		//if (matches.size() == 1) return matches[0]->ast; // exactly one parameter found
		// one or more matching parameters found
		// first, we need to figure out if these are interaction parameters of different polynomial degrees
		// if they are, then all of them are allowed to match
		// if not, match the one with the fewest wildcards
		// TODO: if some have equal numbers of wildcards, choose the first one and warn the user
		std::map<double,const Parameter*> minwilds; // map polynomial degrees to parameters
		bool interactionparam = false;
		bool returnall = true;
		for (auto i = matches.begin(); i != matches.end(); ++i) {
			int wildcount = 0;
			const double curdegree = (*i)->degree;
			const auto array_begin = (*i)->constituent_array.begin();
			const auto array_end = (*i)->constituent_array.end();
			for (auto j = array_begin; j != array_end; ++j) {
				if ((*j)[0] == "*") ++wildcount;
				if ((*j).size() == 2) interactionparam = true; // Binary interaction parameter
				if ((*j).size() == 3) interactionparam = true; // Ternary interaction parameter
			}
			if (minwilds.find(curdegree) == minwilds.end() || wildcount < minwilds[curdegree]->wildcount()) {
				minwilds[curdegree] = (*i);
			}
		}


		// We're fine to return minparam's AST if all polynomial degrees are the same
		// TODO: It seems like it's possible to construct corner cases with duplicate
		// parameters with varying degrees that would confuse this matching.

		//if (minwilds.size() == 1) return minwilds.cbegin()->second->ast;

		if (minwilds.size() > 1 && (!interactionparam)) {
			BOOST_THROW_EXCEPTION(internal_error() << specific_errinfo("Multiple polynomial degrees specified for non-interaction parameters") << ast_errinfo(minwilds.cbegin()->second->ast));
		}

		//if (minwilds.size() > 1 && interactionparam) {
			utree ret_tree;
			if (minwilds.size() != matches.size()) {
				// not all polynomial degrees here are unique
				// it shouldn't be a problem, it should just mean we matched some based on wildcards
				// (this is just here as a note)
			}
			for (auto param = minwilds.begin(); param != minwilds.end(); ++param) {
				const auto array_begin = param->second->constituent_array.begin();
				const auto array_end = param->second->constituent_array.end();
				for (auto j = array_begin; j != array_end; ++j) {
					utree next_term;
					if ((*j).size() == 1) { // Unary parameter (non-interaction)
						next_term = param->second->ast;
					}
					if ((*j).size() == 2) { // Binary interactions
						// get the names of the variables that are interacting
						std::string lhs_var, rhs_var;
						std::stringstream varname1, varname2;
						varname1 << param->second->phasename() << "_" << std::distance(array_begin,j) << "_" << (*j)[0];
						varname2 << param->second->phasename() << "_" << std::distance(array_begin,j) << "_" << (*j)[1];
						lhs_var = varname1.str();
						rhs_var = varname2.str();
						// add to the parameter tree a factor of (y_i - y_j)**k, where k is the degree and i,j are interacting
						next_term = add_interaction_factor(lhs_var, rhs_var, param->second->degree, param->second->ast);
					}
					if ((*j).size() == 3) { // Ternary interactions
						// the order the parameter corresponds to the index of the relevant component in the constituent array
						// should be an integer quantity; left auto here to let other data structures choose type
						auto order = param->second->degree;
						if (order > ((*j).size() - 1)) {
							BOOST_THROW_EXCEPTION(internal_error() << specific_errinfo("Order of ternary interaction parameter is out of bounds"));
						}
						std::stringstream varname;
						varname << param->second->phasename() << "_" << std::distance(array_begin,j) << "_" << (*j)[order];
						std::string varstr(varname.str());
						next_term.push_back("*");
						// TODO: should actually be varstr + Muggianu correction terms for higher-order systems
						next_term.push_back(std::move(varstr));
						next_term.push_back(param->second->ast);
					}
					if (next_term.which() != utree_type::invalid_type) {
						// add next_term to the sum (or make ret_tree equal to first term)
						if (ret_tree.which() != utree_type::invalid_type) {
							utree temp_tree;
							temp_tree.push_back("+");
							temp_tree.push_back(ret_tree);
							temp_tree.push_back(next_term);
							ret_tree.swap(temp_tree);
						}
						else ret_tree = std::move(next_term);
					}
				}
			//}
			BOOST_LOG_SEV(model_log, debug) << "returning: " << ret_tree;
			return ret_tree; // return the parameter tree
		}

		if (minwilds.size() == 0) {
			BOOST_THROW_EXCEPTION(internal_error() << specific_errinfo("Failed to match parameter, but the parameter had already been found"));
		}
	}
	BOOST_LOG_SEV(model_log, debug) << "no parameter found";
	return 0; // no parameter found
}
