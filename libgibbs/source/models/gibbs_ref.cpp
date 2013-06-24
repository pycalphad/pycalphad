/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

	Based on example code from Boost MultiIndex.
	Copyright (c) 2003-2008 Joaquin M Lopez Munoz.
	See http://www.boost.org/libs/multi_index for library home page.

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Model for reference Gibbs energy (from pure elements)

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include <string>
#include <sstream>
#include <set>
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

utree build_Gibbs_ref(
		const std::string &phasename,
		const sublattice_set &subl_set,
		const parameter_set &param_set
		) {
	utree ret_tree;
	sublattice_set_view ssv;
	parameter_set_view psv;
	parameter_set_view psv_subview;

	// Get all the sublattices for this phase
	boost::multi_index::index<sublattice_set,phases>::type::iterator ic0,ic1;
	boost::tuples::tie(ic0,ic1)=get<phases>(subl_set).equal_range(phasename);

	// Construct a view from the iterators
	while (ic0 != ic1) {
		ssv.insert(&*ic0);
		++ic0;
	}

	// Get all the parameters for this phase
	boost::multi_index::index<parameter_set,phase_index>::type::iterator pa_start,pa_end;
	boost::tuples::tie(pa_start,pa_end)=get<phase_index>(param_set).equal_range(phasename);

	// Construct a view from the iterators
	while (pa_start != pa_end) {
		psv.insert(&*pa_start);
		++pa_start;
	}

	// build a subview to the parameters that we are interested in
	boost::multi_index::index<parameter_set_view,type_index>::type::iterator it0, it1;
	std::string scantype = "G";
	boost::tuples::tie(it0,it1)=get<type_index>(psv).equal_range(scantype);

	// Construct a subview from the view
	while (it0 != it1) {
		psv_subview.insert(*it0);
		++it0;
	}

	return permute_site_fractions(ssv, sublattice_set_view(), psv_subview, (int)0);
}

utree permute_site_fractions (
		const sublattice_set_view &total_view, // all sublattices
		const sublattice_set_view &subl_view, // the active sublattice permutation
		const parameter_set_view &param_view,
		const int &sublindex
		) {

	utree ret_tree;
	// Construct a view of just the current sublattice
	boost::multi_index::index<sublattice_set_view,myindex>::type::iterator ic0,ic1;
	boost::tuples::tie(ic0,ic1)=get<myindex>(total_view).equal_range(sublindex);

	if (ic0 == ic1) {
		/* We are at the bottom of the recursive loop
		 * or there's an empty sublattice for some reason (bad).
		 * Use the sublattice permutation to find a matching
		 * parameter, if it exists.
		 */

		return find_parameter_ast(subl_view,param_view);
	}

	for (auto i = ic0; i != ic1; ++i) {
		sublattice_set_view temp_view = subl_view;
		utree current_product;
		utree buildtree;
		temp_view.insert((*i)); // add current species to the view
		/* Construct the expression tree.
		 * Start by building the recursive product of site fractions.
		 */
		current_product.push_back("*");
		// The variable will be represented as a string
		const std::string varname = (*i)->name();
		current_product.push_back(utree(varname));

		utree recursive_term = permute_site_fractions(total_view, temp_view, param_view, sublindex+1);
		if (recursive_term.which() == utree_type::int_type && recursive_term.get<int>() == 0) continue;
		if (recursive_term.which() == utree_type::invalid_type) continue;

		// we only get here for non-zero terms
		current_product.push_back(recursive_term);
		// Contribute this product to the sum
		// Check if we are on the first (or only) term in the sum
		if (i != ic0) {
			buildtree.push_back("+");
			buildtree.push_back(ret_tree);
			buildtree.push_back(current_product);
			ret_tree.swap(buildtree);
		}
		else ret_tree.swap(current_product);
	}

	if (ret_tree.which() == utree_type::invalid_type) ret_tree = utree(0); // no parameter for this term
	return ret_tree;
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
		std::sort(search_config[(*subl_start)->index].begin(), search_config[(*subl_start)->index].end());


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
