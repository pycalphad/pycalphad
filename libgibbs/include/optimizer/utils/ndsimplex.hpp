/*=============================================================================
	Copyright (c) 2012-2014 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// N-Dimensional Simplex Point Generation

#ifndef INCLUDED_NDSIMPLEX
#define INCLUDED_NDSIMPLEX

#include "libgibbs/include/optimizer/halton.hpp"
#include <boost/assert.hpp>
#include <vector>
#include <utility>
#include <algorithm>

struct NDSimplex {
	// Reference: Chasalow and Brand, 1995, "Algorithm AS 299: Generation of Simplex Lattice Points"
	template <typename Func> static inline void lattice (
			const std::size_t point_dimension,
			const std::size_t grid_points_per_major_axis,
			const Func &func
	) {
		BOOST_ASSERT(grid_points_per_major_axis >= 2);
		BOOST_ASSERT(point_dimension >= 1);
		typedef std::vector<double> PointType;
		const double lattice_spacing = 1.0 / (double)grid_points_per_major_axis;
		const PointType::value_type lower_limit = 0;
		const PointType::value_type upper_limit = 1;
		PointType point; // Contains current point
		PointType::iterator coord_find;

		// Special case: 1 component; only valid point is {1}
		if (point_dimension == 1) {
			point.push_back(1);
			func(point);
			return;
		}

		// Initialize algorithm
		point.resize(point_dimension, lower_limit); // Fill with smallest value (0)
		const PointType::const_iterator last_coord = --point.cend();
		coord_find = point.begin();
		*coord_find = upper_limit;
		// point should now be {1,0,0,0....}

		double point_sum;
		do {
			point_sum = *coord_find; // point_sum always includes the active coordinate
			PointType::iterator temp_coord_find = coord_find;
			for (auto i = point.begin(); i != coord_find; ++i) point_sum += *i; // sum all previous coordinates (if any)

			std::advance(temp_coord_find,1);
			if (temp_coord_find != point.end()) {
				*temp_coord_find = upper_limit - point_sum; // Set coord_find+1 to its upper limit (1 - sum of previous coordinates)
				std::advance(temp_coord_find,1);
			}
			for (auto i = temp_coord_find; i != point.end(); ++i) *i = lower_limit; // set remaining coordinates to 0

			func(point); // process the current point

			// coord_find points to the coordinate to be decremented
			coord_find = point.begin();
			while (*coord_find == lower_limit) std::advance(coord_find,1);
			*coord_find -= lattice_spacing;
			if (*coord_find < lattice_spacing) *coord_find = lower_limit; // workaround for floating point issues
		}
		while (coord_find != last_coord || point_sum > 0);
	}
	template <typename Func> static inline void quasirandom_sample (
			const std::size_t point_dimension,
			const std::size_t number_of_points,
			const Func &func
	) {
		// TODO: Add the shuffling part to the Halton sequence. This will help with correlation problems for large N
		// TODO: Default-add the end-members (vertices) of the N-simplex
		for (auto sequence_pos = 1; sequence_pos <= number_of_points; ++sequence_pos) {
			std::vector<double> point;
			double point_sum = 0;
			for (auto i = 0; i < point_dimension; ++i) {
				// Draw the coordinate from an exponential distribution
				// N samples from the exponential distribution, when normalized to 1, will be distributed uniformly
				// on the N-simplex.
				// If X is uniformly distributed, then -LN(X) is exponentially distributed.
				// Since the Halton sequence is a low-discrepancy sequence over (0,1], we substitute it for the uniform distribution
				// This makes this algorithm deterministic and may also provide some domain coverage advantages over a
				// psuedo-random sample.
				double value = -log(halton(sequence_pos,primes[i]));
				point_sum += value;
				point.push_back(value);
			}
			for (auto i = point.begin(); i != point.end(); ++i) *i /= point_sum; // Normalize point to sum to 1
			func(point);
			if (point_dimension == 1) break; // no need to generate additional points; only one feasible point exists for 1-simplex
		}
	}
};

#endif
