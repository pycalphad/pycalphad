/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Model for reference Gibbs energy (from pure elements)

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>

using boost::spirit::utree;
using boost::multi_index_container;
using namespace boost::multi_index;

struct sublattice_entry {
	int index; // sublattice index
	double num_sites; // number of sites
	std::string species; // species name
	sublattice_entry (int index_, double num_sites_, std::string species_) :
		index(index_), num_sites(num_sites_), species(species_) {}
};

struct myindex{};
struct species{};

typedef multi_index_container<
  sublattice_entry,
  indexed_by<
    ordered_non_unique<tag<myindex>,BOOST_MULTI_INDEX_MEMBER(sublattice_entry,int,index)>
    >
> sublattice_set;

utree build_Gibbs_ref(
		Sublattice_Collection::const_iterator subl_start,
		Sublattice_Collection::const_iterator subl_end,
		const evalconditions &sysstate
		) {
	utree ret_tree;
	sublattice_set subl_set;
	int sublcount = std::distance(subl_start, subl_end);

	// Build a structure of all the species in all the sublattices
	for (auto j = subl_start; j != subl_end; ++j) {
		for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
			// Check if this species in this sublattice is on our list of elements to investigate
			if (std::find(sysstate.elements.cbegin(),sysstate.elements.cend(),*k) != sysstate.elements.cend()) {
				int sublindex = std::distance(subl_start,j);
				double sitecount = (*j).stoi_coef;
				std::string spec = (*k);
				subl_set.insert(sublattice_entry(sublindex, sitecount, spec));
			}
		}
	}

	// iterate over all sublattices
	// for each sublattice, snag a current permutation of active species in that sublattice

	return ret_tree;
}

utree permute_site_fractions (const sublattice_set &subl_set, const int &sublindex, const int &sublcount) {
	/* Create a sublattice view class
	 * Add to it as we construct a given permutation and pass it as an argument
	 * This method is GENERAL. We can construct views with multiple species in one sublattice
	 * Eventually we can make the get_parameter code work by checking if a given view is a permutation
	 * of one in the parameter database
	 * If I make the view ordered then I don't even have to check the permutation
	 * I can do a direct comparison
	 */
	if (sublindex == sublcount) {
		// We are at the end of the recursive loop
		// Get the corresponding parameter
		return utree(1);
	}
	// Get all the species in the sublattice of index sublindex
	boost::multi_index::index<sublattice_set,myindex>::type::iterator ic0,ic1;
	boost::tuple::tie(ic0,ic1)=get<myindex>(subl_set).equal_range(sublindex);

	for (auto i = ic0; ic0 != ic1; ++i) {
		// test
	}
	return utree();
}
