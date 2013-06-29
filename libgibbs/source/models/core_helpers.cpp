/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

	Based on example code from Boost MultiIndex.
	Copyright (c) 2003-2008 Joaquin M Lopez Munoz.
	See http://www.boost.org/libs/multi_index for library home page.

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Helper functions for AST-based models

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include <string>
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

sublattice_set build_variable_map(
		const Phase_Collection::const_iterator p_begin,
		const Phase_Collection::const_iterator p_end,
		const evalconditions &conditions,
		std::map<std::string, int> &indices
		) {
	sublattice_set ret_set;

	int indexcount = 0; // counter for variable indices (for optimizer)

	// All phases
	for (auto i = p_begin; i != p_end; ++i) {
		auto const cond_find = conditions.phases.find(i->first);
		if (cond_find->second != PhaseStatus::ENTERED) continue;
		auto subl_start = i->second.get_sublattice_iterator();
		auto subl_end = i->second.get_sublattice_iterator_end();
		std::string phasename = i->first;

		indices[phasename + "_FRAC"] = indexcount; // save index of phase fraction
		// insert fake record for the phase fraction variable at -1 sublattice index

		ret_set.insert(sublattice_entry(-1, indexcount++, 0, phasename, ""));

		// All sublattices
		for (auto j = subl_start; j != subl_end;++j) {
			// All species
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					int sublindex = std::distance(subl_start,j);
					double sitecount = (*j).stoi_coef;
					std::string spec = (*k);
					std::stringstream varname;
					varname << phasename << "_" << sublindex << "_" << spec; // build variable name
					indices[varname.str()] = indexcount; // save index of variable

					ret_set.insert(sublattice_entry(sublindex, indexcount++, sitecount, phasename, spec));
				}
			}
		}
	}
	return ret_set;
}

// count the total number of "mixing" sites in a sublattice set
// non-mixing sites are sublattices with only vacancies in them
double count_mixing_sites(const sublattice_set_view &ssv) {
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

// Normalize by the total number of mixing sites
void normalize_utree(utree &input_tree, const sublattice_set_view &ssv) {
	utree temp;
	temp.push_back("/");
	temp.push_back(input_tree);
	temp.push_back(count_mixing_sites(ssv));
	input_tree.swap(temp);
}

utree find_parameter_ast(const sublattice_set_view &subl_view, const parameter_set_view &param_view) {
	// TODO: check degree of parameters (for interaction parameters)
	// TODO: poorly defined behavior for multiple matching parameters
	// one idea: keep a vector of matching parameters; at the end, return the one with fewest *'s
	// if "*" count is equal, throw an exception or choose the first one
	utree ret_tree;
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
			if (!(
				(*array_iter) == search_config[std::distance(array_begin,array_iter)])
				|| (firstelement == "*")
				) {
				isvalid = false;
				break;
			}
			++array_iter;
		}
		if (isvalid) {
			// We found a valid parameter, return its abstract syntax tree
			ret_tree = (*param_iter)->ast;
			break;
		}

		++param_iter;
	}

	return ret_tree;
}
