/*=============================================================================
	Copyright (c) 2012-2014 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// N-Dimensional Simplex Point Generation

#ifndef INCLUDED_NDSIMPLEX
#define INCLUDED_NDSIMPLEX

#include "libgibbs/include/optimizer/halton.hpp"
#include "libgibbs/include/utils/primes.hpp"
#include <boost/assert.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <vector>

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
		PointType::iterator coord_find; // corresponds to 'j' in Chasalow and Brand

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

	// TODO: Is there a way to do this without all the copying?
	static inline std::vector<std::vector<double>> lattice_complex(
			const std::vector<std::size_t> &components_in_sublattices,
			const std::size_t grid_points_per_major_axis
			) {
		using boost::math::factorial;
		typedef std::vector<double> PointType;
		typedef std::vector<PointType> PointCollection;
		std::vector<PointCollection> point_lattices; //  Simplex lattices for each sublattice
		std::vector<PointType> points; // The final return points (combination of all simplex lattices)
		std::size_t expected_points = 1;
		point_lattices.reserve(components_in_sublattices.size());

		for (auto i = components_in_sublattices.cbegin(); i != components_in_sublattices.cend(); ++i) {
			PointCollection returned_points;
			const unsigned int m = grid_points_per_major_axis - 2; // number of evenly spaced values _between_ 0 and 1
			const unsigned int q = *i; // number of components
			const std::size_t lattice_points =
					static_cast<std::size_t>(factorial<double>(q+m-1) / (factorial<double>(m)*factorial<double>(q-1)));
			expected_points *= lattice_points;
			returned_points.reserve(lattice_points); // allocate this memory for convenience

			auto point_add = [&returned_points](std::vector<double> &address) {
				returned_points.push_back(address);
			};
			lattice(*i,grid_points_per_major_axis, point_add);
			BOOST_ASSERT(lattice_points == returned_points.size());
			point_lattices.push_back(returned_points);
		}

		// Now, take all the combinations of points
		points.reserve(expected_points);

		for (auto subl = point_lattices.begin(); subl != point_lattices.end(); ++subl) {
			std::vector<double> point;
		}

		return points;
	}

	// Reference for Halton sequence: Hess and Polak, 2003.
	// Reference for uniformly sampling the simplex: Any text on the Dirichlet distribution
	template <typename Func> static inline void quasirandom_sample (
			const std::size_t point_dimension,
			const std::size_t number_of_points,
			const Func &func
	) {
		BOOST_ASSERT(point_dimension < primes_size()); // No realistic problem should ever violate this
		// TODO: Add the shuffling part to the Halton sequence. This will help with correlation problems for large N
		// TODO: Default-add the end-members (vertices) of the N-simplex
		for (auto sequence_pos = 1; sequence_pos <= number_of_points; ++sequence_pos) {
			std::vector<double> point;
			double point_sum = 0;
			for (auto i = 0; i < point_dimension; ++i) {
				// Draw the coordinate from an exponential distribution
				// N samples from the exponential distribution, when normalized to 1, will be distributed uniformly
				// on a facet of the N-simplex.
				// If X is uniformly distributed, then -LN(X) is exponentially distributed.
				// Since the Halton sequence is a low-discrepancy sequence over [0,1], we substitute it for the uniform distribution
				// This makes this algorithm deterministic and may also provide some domain coverage advantages over a
				// psuedo-random sample.
				double value = -log(halton(sequence_pos,primes[i]));
				point_sum += value;
				point.push_back(value);
			}
			for (auto i = point.begin(); i != point.end(); ++i) *i /= point_sum; // Normalize point to sum to 1
			func(point);
			if (point_dimension == 1) break; // no need to generate additional points; only one feasible point exists for 0-simplex
		}
	}
};

#endif
