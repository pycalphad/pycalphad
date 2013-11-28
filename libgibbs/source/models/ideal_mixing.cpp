/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// model for ideal entropy of mixing

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include "libgibbs/include/models.hpp"
#include "libtdb/include/logging.hpp"
#include <string>
#include <limits>
#include <sstream>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/tuple/tuple.hpp>

using boost::spirit::utree;
using boost::multi_index_container;
using namespace boost::multi_index;

// helper function to help x*ln(x) avoid domain errors when x <= 0

void IdealMixingModel::protect_domain(const std::string &varname, utree &input_tree) {
	utree temp_tree;
	temp_tree.push_back("@"); // constraint operator
	temp_tree.push_back(varname); // constrained variable
	temp_tree.push_back(-std::numeric_limits<double>::max()); // low limit
	temp_tree.push_back(1e-20); // high limit
	temp_tree.push_back(varname); // x*log(x) ~ x for small x
	temp_tree.push_back("@"); // constraint operator
	temp_tree.push_back(varname);
	temp_tree.push_back(1e-20); // low limit
	temp_tree.push_back(std::numeric_limits<double>::max()); // maximum value
	temp_tree.push_back(input_tree);
	input_tree.swap(temp_tree);
}

// helper function to produce utrees of form x*ln(x)
utree IdealMixingModel::make_xlnx(const std::string &varname) {
	utree ln_tree, ret_tree;
	// make ln(x)
	ln_tree.push_back("LN");
	ln_tree.push_back(varname);
	// make x*ln(x)
	ret_tree.push_back("*");
	ret_tree.push_back(varname);
	ret_tree.push_back(ln_tree);
	protect_domain(varname, ret_tree);
	return ret_tree;
}

IdealMixingModel::IdealMixingModel(
		const std::string &phasename,
		const sublattice_set &subl_set
		) : EnergyModel(phasename, subl_set) {
	BOOST_LOG_NAMED_SCOPE("IdealMixingModel::IdealMixingModel");
	// TODO: add intelligence to detect single-species sublattices (no mixing contribution)
	utree work_tree, gas_const_product;
	sublattice_set_view ssv;
	int curindex = 0;
	logger opt_log(journal::keywords::channel = "optimizer");
	// Get all the sublattices for this phase
	boost::multi_index::index<sublattice_set,phases>::type::iterator ic0,ic1;
	boost::tuples::tie(ic0,ic1)=get<phases>(subl_set).equal_range(phasename);

	// Generate a subview which excludes the "-1" fake sublattice index
	if (ic0 == ic1) BOOST_LOG_SEV(opt_log, critical) << "Sublattice set in ideal mixing model empty!";
	while (ic0 != ic1) {
		BOOST_LOG_SEV(opt_log, debug) << "ic0: " << ic0->name();
		ssv.insert(&*ic0);
		++ic0;
	}
	boost::multi_index::index<sublattice_set_view,myindex>::type::iterator s_start = get<myindex>(ssv).lower_bound(curindex);
	boost::multi_index::index<sublattice_set_view,myindex>::type::iterator s_end = get<myindex>(ssv).upper_bound(curindex);
	boost::multi_index::index<sublattice_set_view,myindex>::type::iterator s_final_end = get<myindex>(ssv).end();

	if (s_start == s_final_end || s_start == s_end) BOOST_LOG_SEV(opt_log, critical) << "Sublattice set view in ideal mixing model empty!";
	while (s_start != s_final_end) {
		utree subl_tree, temptree_loop;

		// Loop through all species in current sublattice
		for (auto i = s_start; i != s_end; ++i) {
			utree current_product;
			if (i != s_start) {
				current_product.push_back("+");
				current_product.push_back(subl_tree);
				current_product.push_back(make_xlnx((*i)->name()));
				subl_tree.swap(current_product);
			}
			else subl_tree = make_xlnx((*i)->name());
		}

		// Multiply subl_tree by the number of sites in this sublattice

		temptree_loop.push_back("*");
		temptree_loop.push_back((*s_start)->num_sites);
		temptree_loop.push_back(subl_tree);
		subl_tree.swap(temptree_loop);
		temptree_loop.clear();


		if (curindex > 0) {
			// This is not the first sublattice
			temptree_loop.push_back("+");
			temptree_loop.push_back(work_tree);
			temptree_loop.push_back(subl_tree);
			work_tree.swap(temptree_loop);
		}
		else work_tree.swap(subl_tree);

		// Advance to next sublattice
		++curindex;
		s_start = get<myindex>(ssv).lower_bound(curindex);
		s_end = get<myindex>(ssv).upper_bound(curindex);
	}

	// add R*T as a product in front and normalize by the number of sites
	model_ast.push_back("*");
	model_ast.push_back("T");
	gas_const_product.push_back("*");
	gas_const_product.push_back(SI_GAS_CONSTANT);
	gas_const_product.push_back(work_tree);
	model_ast.push_back(gas_const_product);
	normalize_utree(model_ast, ssv);
}
