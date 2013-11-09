/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// header file for Gibbs energy optimizer
#ifndef INCLUDED_OPTIMIZER
#define INCLUDED_OPTIMIZER

#include "libtdb/include/structure.hpp"
#include <vector>
#include <boost/tuple/tuple.hpp>

struct vector_map {
	// int,int means begin_index, end_index
	// std::string is name of species
	typedef std::vector<std::pair<int,int>> index_pairs;
	std::vector<boost::tuple<int,int,Phase_Collection::const_iterator>> phasefrac_iters;
	// phase->sublattice->species[name]->pair(index,phase_iter)
	std::vector<std::vector<std::map<std::string,std::pair<int,Phase_Collection::const_iterator>>>> sitefrac_iters;
};

#endif
