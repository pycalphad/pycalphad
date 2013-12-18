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
#include "libgibbs/include/optimizer/halton.hpp"


namespace Optimizer {

// TODO: Should this be a member function of GibbsOpt?
// The function calling LocateMinima definitely should be at least (needs access to all CompositionSets)
// LocateMinima finds all of the minima for a given phase's Gibbs energy
// In addition to allowing us to choose a better starting point, this will allow for automatic miscibility gap detection
void LocateMinima(std::unique_ptr<CompositionSet> const &phase, const int depth) {
	// Because the grid is uniform, we can assume that each point is the center of an N-cube
	// of width max_extent-min_extent. Boundary points are a special case.
	// Do whatever normalization is needed to put them in the feasible region
	//
	// EZD Global Minimization (Emelianenko et al., 2006)
	// For depth = 1: FIND CONCAVITY REGIONS
	// (1) Sample some points on the domain using NDGrid (probably separate function)
	// (2) Calculate the Lagrangian Hessian (L'') for all sampled points
	// (3) Verify that all diagonal elements of L'' are strictly positive; if not, remove this point from consideration
	//     NOTE: This is a necessary but not sufficient condition that a matrix be positive definite, and it's easy to check
	//     Reference: Carlen and Carvalho, 2007, p. 148, Eq. 5.12
	// NOTE: For this calculation we consider only the linear constraints for an isolated phase (e.g., site fraction balances)
	// (4) Save all points for which the Lagrangian Hessian is positive definite in the null space of the constraint gradient matrix
	//        NOTE: This is the two-sided projected Hessian method (Nocedal and Wright, 2006, ch. 12.4, p.349)
	//        TODO: But how do I choose the Langrange multipliers for all the constraints? Can I calculate them?
	//    (a) Form matrix A, the Jacobian of active constraints (constraint gradient matrix)
	//    (b) Perform QR factorization of transpose(A)
	//    (c) Set Z = Q2, which is defined by (TODO: still not clear on how to do this) Eq. 12.71, p. 349 of Nocedal and Wright, 2006
	//    (d) Set Hproj = transpose(Z)*(L'')*Z
	//    (e) Attempt a Cholesky factorization of Hproj; will only succeed if matrix is positive definite
	//    (f) If it succeeds, save this point; else, remove it
	// (5) For each saved point, send to next depth
	// For depth > 1: FIND MINIMA
	// (1) Sample some points on the domain using NDGrid (probably separate function)
	// (2) Calculate the Lagrangian gradient (L') for all sampled points
	// (3) Find the point (z) with the minimum magnitude of L'
	// (4) If that magnitude is less than some defined epsilon, or we've hit max_depth, return z as a minimum
	//     Else use N-cube assumption to define new domain around z and send to next depth (return minima from that)

}

}
