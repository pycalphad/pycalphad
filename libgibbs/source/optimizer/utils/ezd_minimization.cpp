/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// subroutines for EZD global minimization
// Reference: Maria Emelianenko, Zi-Kui Liu, and Qiang Du.
// "A new algorithm for the automation of phase diagram calculation." Computational Materials Science 35.1 (2006): 61-74.

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/utils/ezd_minimization.hpp"
#include "libgibbs/include/compositionset.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/halton.hpp"
#include "libgibbs/include/optimizer/utils/ndgrid.hpp"
#include "libgibbs/include/utils/cholesky.hpp"
#include "libgibbs/include/utils/qr.hpp"
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <algorithm>

namespace Optimizer {

// TODO: Should this be a member function of GibbsOpt?
// The function calling LocateMinima definitely should be at least (needs access to all CompositionSets)
// LocateMinima finds all of the minima for a given phase's Gibbs energy
// In addition to allowing us to choose a better starting point, this will allow for automatic miscibility gap detection
void LocateMinima(CompositionSet const &phase, sublattice_set const &sublset, const std::size_t depth) {
	constexpr const std::size_t grid_points_per_axis = 5; // TODO: make this user-configurable
	// Because the grid is uniform, we can assume that each point is the center of an N-cube
	// of width max_extent-min_extent. Boundary points are a special case.
	// Drop points outside the feasible region.
	//
	// EZD Global Minimization (Emelianenko et al., 2006)
	// For depth = 1: FIND CONCAVITY REGIONS
	if (depth == 1) {
		std::vector<std::vector<double> > points;

		// Get all the sublattices for this phase
		boost::multi_index::index<sublattice_set,phases>::type::iterator ic0,ic1;
		boost::tuples::tie(ic0,ic1)=boost::multi_index::get<phases>(sublset).equal_range(phase.name());

		// (1) Sample some points on the domain using NDGrid
		// TODO: This is going to generate a lot of infeasible points
		// A better implementation would handle the sublattice internal degrees of freedom separately
		auto point_add = [&points](std::vector<double> &address) {
				std::cout << "adding point [";
				for (auto i = address.begin(); i != address.end(); ++i) {
					std::cout << *i;
					if (std::distance(i,address.end()) > 1) std::cout << ",";
				}
				std::cout << "]" << std::endl;
				points.push_back(address);
		};
		NDGrid::sample(0, 1, std::distance(ic0,ic1), grid_points_per_axis, point_add);
		// (2) Calculate the Lagrangian Hessian (L'') for all sampled points
		// NOTE: For this calculation we consider only the linear constraints for an isolated phase (e.g., site fraction balances)
		// (3) Save all points for which the Lagrangian Hessian is positive definite in the null space of the constraint gradient matrix
		//        NOTE: This is the two-sided projected Hessian method (Nocedal and Wright, 2006, ch. 12.4, p.349)
		//        But how do I choose the Lagrange multipliers for all the constraints? Can I calculate them?
		//        The answer is that, because the constraints are linear, there is no constraint contribution to the Hessian.
		//        That means that the Hessian of the Lagrangian is just the Hessian of the objective function.
		//    (a) Form matrix A (m x n), the Jacobian of active constraints (constraint gradient matrix)
		//    (b) Compute the full QR decomposition of transpose(A)
		//    (c) Copy the last m-n columns of Q into Z (related to the bottom m-n rows of R which should all be zero)
		//        Reference: Eq. 12.71, p. 349 of Nocedal and Wright, 2006
		//    (d) Set Hproj = transpose(Z)*(L'')*Z
		//    (e) Verify that all diagonal elements of Hproj are strictly positive; if not, remove this point from consideration
		//        NOTE: This is a necessary but not sufficient condition that a matrix be positive definite, and it's easy to check
		//        Reference: Carlen and Carvalho, 2007, p. 148, Eq. 5.12
		//    (f) Attempt a Cholesky factorization of Hproj; will only succeed if matrix is positive definite
		//    (g) If it succeeds, save this point; else, remove it
		// (4) For each saved point, send to next depth
	}
	// For depth > 1: FIND MINIMA
	else if (depth > 1) {
		// (1) Sample some points on the domain using NDGrid (probably separate function)
		// (2) Calculate the Lagrangian gradient (L') for all sampled points
		// (3) Find the point (z) with the minimum magnitude of L'
		// (4) If that magnitude is less than some defined epsilon, or we've hit max_depth, return z as a minimum
		//     Else use N-cube assumption to define new domain around z and send to next depth (return minima from that)
	}
	else {
		// invalid depth; throw exception
	}
}

}
