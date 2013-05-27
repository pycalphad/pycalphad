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
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include "libgibbs/include/models.hpp"
#include <string>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/tuple/tuple.hpp>

using boost::spirit::utree;
using boost::multi_index_container;
using namespace boost::multi_index;
using namespace models;

sublattice_set build_variable_map(
		const Phase_Collection::const_iterator p_begin,
		const Phase_Collection::const_iterator p_end,
		const evalconditions &conditions
		) {
	sublattice_set ret_set;

	// All phases
	for (auto i = p_begin; i != p_end; ++i) {
		auto subl_start = i->second.get_sublattice_iterator();
		auto subl_end = i->second.get_sublattice_iterator_end();
		// All sublattices
		for (auto j = subl_start; j != subl_end;++j) {
			// All species
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					int sublindex = std::distance(subl_start,j);
					double sitecount = (*j).stoi_coef;
					std::string spec = (*k);
					std::string phasename = i->first;
					ret_set.insert(sublattice_entry(sublindex, sitecount, phasename, spec));
				}
			}
		}
	}
	return ret_set;
}

utree build_Gibbs_ref(
		const std::string &phasename,
		const sublattice_set &subl_set
		) {
	utree ret_tree;
	sublattice_set_view ssv;

	// Get all the sublattices for this phase
	boost::multi_index::index<sublattice_set,phases>::type::iterator ic0,ic1;
	boost::tuples::tie(ic0,ic1)=get<phases>(subl_set).equal_range(phasename);

	// Construct a view from the iterators
	while (ic0 != ic1) {
		ssv.insert(*ic0);
		++ic0;
	}
	return permute_site_fractions(ssv, sublattice_set_view(), (int)0);
}

utree permute_site_fractions (
		const sublattice_set_view &total_view, // all sublattices
		const sublattice_set_view &subl_view, // the active sublattice permutation
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
		return utree(1);
	}

	for (auto i = ic0; ic0 != ic1; ++i) {
		sublattice_set_view temp_view = subl_view;
		utree current_product;
		utree buildtree;
		temp_view.insert((*i)); // add ptr to current species to the view

		/* Construct the expression tree.
		 * Start by building the recursive product of site fractions.
		 */
		current_product.push_back("*");
		// The variable will be represented as a pointer
		current_product.push_back(utree(&*i));
		current_product.push_back(
				permute_site_fractions(total_view, temp_view, sublindex+1)
				);

		// Contribute this product to the sum
		buildtree.push_back("+");
		buildtree.push_back(current_product);
		buildtree.push_back(ret_tree);
		ret_tree.swap(buildtree);
	}
	return ret_tree;
}
