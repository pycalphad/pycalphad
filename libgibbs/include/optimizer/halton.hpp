/*=============================================================================
	Copyright (c) 2012-2014 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// implementation of Halton sequences

#ifndef INCLUDED_HALTON
#define INCLUDED_HALTON

#include "libgibbs/include/utils/primes.hpp"
#include <cmath>

inline double halton(unsigned int index, unsigned int base) {
	double result = 0;
	double f = 1.0 / (double)base;
	unsigned int i = index;
	while(i > 0) {
		result = result + f * (i % base);
		i = floor(i / base);
		f = f / base;
	}
	return result;
};

#endif
