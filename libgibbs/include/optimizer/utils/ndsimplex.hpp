/*=============================================================================
	Copyright (c) 2012-2014 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// N-Dimensional Simplex Lattice

#ifndef INCLUDED_NDSIMPLEX
#define INCLUDED_NDSIMPLEX

#include "libgibbs/include/optimizer/halton.hpp"
#include <vector>
#include <utility>
#include <algorithm>

struct NDSimplex {
	template <typename Func> static void sample(
			const std::vector<std::pair<double,double> > &extents,
			const double grid_points_per_major_axis,
			const Func &func,
			std::vector<double>& address,
			double sum_of_address = 0) {
		if (address.size() == extents.size()) {
			// terminating condition; address is complete
			func(address);
		}
		else {
			const double max_extent = std::min(extents[address.size()].second, 1 - sum_of_address);
			const double min_extent = extents[address.size()].first;
			double step = (max_extent - min_extent) / grid_points_per_major_axis;
			for (auto j = 0; j <= grid_points_per_major_axis; ++j) {
				double location = step*j + min_extent;
				if (sum_of_address + location >= 1) {
					// Force the last coordinate onto the simplex face
					location = std::max(1 - sum_of_address,0.);
					address.push_back(location);
					// Make the remaining coordinates all zero
					std::size_t added_elements = extents.size() - address.size();
					while (address.size() < extents.size()) address.push_back(0);
					// Perform callback
					func(address);
					// Remove extra elements
					while (added_elements--) address.pop_back();
					// Remove the element we forced onto the simplex face
					address.pop_back();
					// No more coordinates to handle in this iteration
					break;
				}
				else {
					address.push_back(location);
					// recursive step
					NDSimplex::sample(extents, grid_points_per_major_axis, func, address, sum_of_address + location);
					address.pop_back(); // remove the element we just added (this way avoids copying)
					if (step == 0) break; // don't generate duplicate points
				}
			}
		}
	}
	template <typename Func> static inline void sample(
			const std::vector<std::pair<double,double> > &extents,
			const double grid_points_per_major_axis,
			const Func &func) {
		std::vector<double> address;
		NDSimplex::sample(extents, grid_points_per_major_axis, func, address, 0);
	}
	template <typename Func> static inline void quasirandom_sample (
			const unsigned int point_dimension,
			const unsigned int number_of_points,
			const Func &func
	) {
		// TODO: add the shuffling part
		for (auto sequence_pos = 1; sequence_pos <= point_dimension; ++sequence_pos) {
			std::vector<double> point;
			double point_sum = 0;
			for (auto i = 0; i < number_of_points; ++i) {
				// Give the coordinate an exponential distribution
				double value = -log(halton(sequence_pos,primes[i]));
				point_sum += value;
				point.push_back(value);
			}
			for (auto i = point.begin(); i != point.end(); ++i) *i /= point_sum; // Normalize point to sum to 1
			func(point);
		}
	}
};

#endif
