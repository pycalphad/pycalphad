/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// implementation of shuffled Halton sequences for quasi-random point sampling
// See Hess and Polak, 2003.

#include "libgibbs/include/libgibbs_pch.hpp"
#include <math.h>
#include <vector>
#include <boost/multi_array.hpp>
#include "libgibbs/include/optimizer/halton.hpp"

double halton(unsigned int index, unsigned int base) {
	double result = 0;
	double f = 1 / (double)base;
	unsigned int i = index;
	while(i > 0) {
		result = result + f * (i % base);
		i = floor(i / base);
		f = f / base;
	}
	return result;
};

point_list point_sample (
		const unsigned int varcount,
		const unsigned int numpts
		) {
	// TODO: add the shuffling part
	point_list points(boost::extents[varcount][numpts-1]);
	for (auto i = 0; i < varcount; ++i) {
		for (auto sequence_pos = 1; sequence_pos <= numpts; ++sequence_pos) {
			points[sequence_pos-1][i] = halton(sequence_pos,primes[i]);
			points[sequence_pos-1][varcount] = 0; // extra user-defined value associated with pt
		}
		std::cout << std::endl;
	}
	for (auto i = 0; i < numpts; ++i) {
		//std::cout << "point: ";
		for (auto j = 0; j < varcount; ++j) {
			//std::cout << points[i][j] << " ";
		}
		//std::cout << std::endl;
	}
	return points;
}
