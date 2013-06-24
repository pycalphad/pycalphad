/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// model for ideal entropy of mixing

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include "libgibbs/include/models.hpp"
#include <string>
#include <sstream>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/tuple/tuple.hpp>

using boost::spirit::utree;
using boost::multi_index_container;
using namespace boost::multi_index;

// helper function to produce utrees of form x*ln(x)
utree make_xlnx(const std::string &varname) {
	utree ln_tree, ret_tree;
	// make ln(x)
	ln_tree.push_back("LN");
	ln_tree.push_back(varname);
	// make x*ln(x)
	ret_tree.push_back("*");
	ret_tree.push_back(varname);
	ret_tree.push_back(ln_tree);
	return ret_tree;
}

utree build_ideal_mixing_entropy(
		const std::string &phasename,
		const sublattice_set &subl_set
		) {
	// TODO: add intelligence to detect single-species sublattices (no mixing contribution)
	utree ret_tree, temptree, gas_const_product;
	sublattice_set_view ssv;

	// Get all the sublattices for this phase
	boost::multi_index::index<sublattice_set,phases>::type::iterator ic0,ic1;
	boost::tuples::tie(ic0,ic1)=get<phases>(subl_set).equal_range(phasename);

	// Generate a subview which excludes the "-1" fake sublattice index
	while (ic0 != ic1) {
		ssv.insert(&*ic0);
		++ic0;
	}
	boost::multi_index::index<sublattice_set_view,myindex>::type::iterator s_start = get<myindex>(ssv).lower_bound(0);
	boost::multi_index::index<sublattice_set_view,myindex>::type::iterator s_end = get<myindex>(ssv).end();

	for (auto i = s_start; i != s_end; ++i) {
		utree current_product;
		if (i != s_start) {
			current_product.push_back("+");
			current_product.push_back(ret_tree);
			current_product.push_back(make_xlnx((*i)->name()));
			ret_tree.swap(current_product);
		}
		else ret_tree = make_xlnx((*i)->name());
	}

	// add R*T as a product in front
	temptree.push_back("*");
	temptree.push_back("T");
	gas_const_product.push_back("*");
	gas_const_product.push_back(SI_GAS_CONSTANT);
	gas_const_product.push_back(ret_tree);
	temptree.push_back(gas_const_product);

	return temptree;
}
