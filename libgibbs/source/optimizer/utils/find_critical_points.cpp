/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// adaptive search method for finding critical points (EZD method)
// See Emelianenko, Liu, and Du (2006)

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libtdb/include/structure.hpp"
#include "libtdb/include/conditions.hpp"
#include <libgibbs/include/optimizer/halton.hpp>
#include <boost/multi_array.hpp>

typedef boost::multi_array<double,2> domain;
point_list find_critical_points (
		const Phase_Collection::const_iterator phase_iter,
		const evalconditions &conditions,
		domain V,
		unsigned int iter_index
		) {
	const unsigned int N = 20;
	const double eps_tol = 1e-6;
	const unsigned int iter_max = 100;
	const auto subl_begin = phase_iter->second.get_sublattice_iterator();
	const auto subl_end = phase_iter->second.get_sublattice_iterator_end();
	const unsigned int subl_count = std::distance(subl_begin, subl_end);
	unsigned int varcount = 0;
	point_list sublattice_extents(boost::extents[subl_count][2]);

	for (auto j = phase_iter->second.get_sublattice_iterator();
			j != phase_iter->second.get_sublattice_iterator_end(); ++j) {
		const unsigned int subl_index = std::distance(subl_begin,j);
		sublattice_extents[subl_index][0] = varcount;
		for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
			// Check if this species in this sublattice is on our list of elements
			if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) !=
					conditions.elements.cend()) {
				// This site matches one of our elements under investigation
				++varcount;
			}
		}
		sublattice_extents[subl_index][1] = varcount;
	}

	point_list samples(boost::extents[N][varcount]);
	samples = point_sample(varcount,N);
	while (iter_index <= iter_max) {
	}
	}
	return V;
}
