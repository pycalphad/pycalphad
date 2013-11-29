/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition for a function to find a naive starting point

#include "libgibbs/include/optimizer/utils/startingpoint_naive.hpp"

using boost::multi_index_container;
using namespace boost::multi_index;


// Set all site fractions to 1/X, where X is the number of entered species in that sublattice
// Set all phase fractions to 1/Y, where Y is the number of entered phases
template <typename Y = int, typename T = double> std::map<Y,T> get_startingpoint_naive(const sublattice_set &total_view)  {
	std::map<Y,T> retmap;
	boost::multi_index::index<sublattice_set,myindex>::type::const_iterator iter,end;

	// System structure iterators
	iter = get<myindex>(total_view).begin();
	end = get<myindex>(total_view).end();

	// Iterate through the whole system
	for (; iter != end; ++iter) {
		const int sublindex = ((sublattice_set::const_iterator)iter)->index;
		const Y opt_index = ((sublattice_set::const_iterator)iter)->opt_index;
		if (sublindex == -1) {
			// This is a phase fraction; count total number of entered phases
			const size_t phasecount = total_view.get<myindex>().count(sublindex);
			retmap[opt_index] = (T)1 / (T)phasecount;
		}
		else {
			const std::string phasename = ((sublattice_set::const_iterator)iter)->phase;
			// Count number of entered species in the current sublattice
			const size_t speccount = total_view.get<phase_subl>().count(boost::make_tuple(phasename, sublindex));
			retmap[opt_index] = (T)1 / (T)speccount;
		}
	}
	return retmap;
}
