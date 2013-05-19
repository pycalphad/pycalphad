/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Hessian approximation by finite differences for G''(x)
// Used for curvature analysis in point sampling procedure

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/optimizer.hpp"
#include <map>

/* This really only needs to be good enough to get us the sign.
 * Ipopt has a more sophisticated Hessian approximation built-in.
 * We also only need to calculate the diagonal term here.
 */
double hessian_approx(
		const sublattice_vector::const_iterator subl_start,
		const sublattice_vector::const_iterator subl_end,
		const Phase_Collection::const_iterator phase_iter,
		const evalconditions &conditions,
		const int &sublindex,
		const std::string &specname
) {
	const double step = 1e-6;
	double first = get_Gibbs_deriv(subl_start, subl_end, phase_iter, conditions, sublindex, specname);

	// Copy and modify the sublattice_vector
	// There's probably a better way to do this...
	sublattice_vector subls_vec;
	for (auto i = subl_start; i != subl_end; ++i) {
		std::map<std::string,double> subl_map;
		for (auto j = (*i).begin(); j != (*i).end(); ++j) {
			if (sublindex == std::distance(subl_start,i) && j->first == specname) {
				subl_map[j->first] = j->second + step; // Modify with step
			}
			else subl_map[j->first] = j->second;
		}
		subls_vec.push_back(subl_map);
	}
	sublattice_vector::const_iterator step_start = subls_vec.cbegin();
	sublattice_vector::const_iterator step_end = subls_vec.cend();

	double second = get_Gibbs_deriv(step_start, step_end, phase_iter, conditions, sublindex, specname);

	return (second - first) / step;
}
