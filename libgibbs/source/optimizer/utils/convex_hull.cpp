/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Calculate convex hull using Qhull / libqhullcpp
// All interfacing with the library will be done in this module

#include "libgibbs/include/optimizer/utils/convex_hull.hpp"

#include "external/libqhullcpp/QhullError.h"
#include "external/libqhullcpp/QhullQh.h"
#include "external/libqhullcpp/QhullFacet.h"
#include "external/libqhullcpp/QhullFacetList.h"
#include "external/libqhullcpp/QhullLinkedList.h"
#include "external/libqhullcpp/QhullVertex.h"
#include "external/libqhullcpp/Qhull.h"

using orgQhull::Qhull;
using orgQhull::QhullError;
using orgQhull::QhullFacet;
using orgQhull::QhullFacetList;
using orgQhull::QhullQh;
using orgQhull::QhullVertex;
using orgQhull::QhullVertexSet;

namespace Optimizer { namespace details {
    // Modified QuickHull algorithm using d-dimensional Beneath-Beyond
    // Reference: N. Perevoshchikova, et al., 2012, Computational Materials Science.
    // "A convex hull algorithm for a grid minimization of Gibbs energy as initial step 
    //    in equilibrium calculations in two-phase multicomponent alloys"
    void lower_convex_hull ( const std::vector<std::vector<double>> &points ) {
        Qhull qhull;
    }
} // namespace details
} // namespace Optimizer