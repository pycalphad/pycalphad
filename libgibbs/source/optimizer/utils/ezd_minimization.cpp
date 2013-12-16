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
void LocateMinima(std::unique_ptr<CompositionSet> const &phase, const int depth) {
	// Because the grid is uniform, we can assume that each point is the center of an N-cube
	// of width max_extent-min_extent. Boundary points are a special case.
	// Do whatever normalization is needed to put them in the feasible region
	//
	// EZD Global Minimization (Emelianenko et al., 2006)
	// For initial depth (1): FIND CONCAVITY REGIONS
	//	  Sample some points on the domain using NDGrid (probably separate function)
	//    Calculate G'' for all sampled points
	//    Save all points for which G'' > 0
	//    For each saved point, send to next depth
	// For depth > 1: FIND MINIMA
	//    Sample some points on the domain using NDGrid (probably separate function)
	//    Calculate G' for all sampled points
	//    Find the point (z) with the minimum magnitude of G'
	//    If that magnitude is less than some defined epsilon, or we've hit max_depth, return z as a minimum
	//    Else use N-cube assumption to define new domain around z and send to next depth (return minima from that)

}

}
