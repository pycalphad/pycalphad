/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// N-Dimensional uniform grid

#ifndef INCLUDED_NDGRID
#define INCLUDED_NDGRID

#include <vector>

struct NDGrid {
	template <typename Func> static void sample(
			const double min_extent,
			const double max_extent,
			const std::size_t dimension,
			const double grid_points_per_major_axis,
			const Func &func,
			std::vector<double>& address) {
		if (address.size() == dimension) {
			// terminating condition; address is complete
			func(address);
		}
		else {
			double step = (max_extent - min_extent) / grid_points_per_major_axis;
			for (auto j = 0; j <= grid_points_per_major_axis; ++j) {
				double location = step*j + min_extent;
				address.push_back(location);
				// recursive step
				NDGrid::sample(min_extent, max_extent, dimension, grid_points_per_major_axis, func, address);
				address.pop_back(); // remove the element we just added (this way avoids copying)
			}
		}
	}
	template <typename Func> static inline void sample(
			const double min_extent,
			const double max_extent,
			const std::size_t dimension,
			const double grid_points_per_major_axis,
			const Func &func) {
		std::vector<double> address;
		NDGrid::sample(min_extent, max_extent, dimension, grid_points_per_major_axis, func, address);
	}
};

#endif
