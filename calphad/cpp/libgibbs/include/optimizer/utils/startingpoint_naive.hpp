/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// declaration for a function to find a naive starting point

#ifndef INCLUDED_STARTINGPOINT_NAIVE
#define INCLUDED_STARTINGPOINT_NAIVE

#include "libgibbs/include/models.hpp"
#include <map>

template <typename Y = int, typename T = double> std::map<Y,T> get_startingpoint_naive(const sublattice_set &total_view);

#endif
