/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// implementation of shuffled Halton sequences for quasi-random point sampling
// See Hess and Polak, 2003.

#include "libgibbs/include/libgibbs_pch.hpp"
#include <vector>
#include <boost/multi_array.hpp>
#include "libgibbs/include/optimizer/halton.hpp"

point_list point_sample (
		const unsigned int varcount,
		const unsigned int numpts
		) {
	// TODO: add the shuffling part
	point_list points(boost::extents[varcount][numpts-1]);
	for (auto i = 0; i < varcount; ++i) {
		double result = 0;
		double f = 1 / (double)primes[i];
		for (auto sequence_pos = 1; sequence_pos <= numpts; ++sequence_pos) {
			result = result + f * (sequence_pos % primes[i]);
			points[sequence_pos-1][i] = result;
			points[sequence_pos-1][varcount] = 0; // extra user-defined value associated with pt
			f = f / (double)primes[i];
		}
	}
	return points;
}
