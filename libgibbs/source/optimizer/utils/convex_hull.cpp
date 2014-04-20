/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Calculate convex hull using Qhull / libqhullcpp
// All interfacing with the library will be done in this module

#include "libgibbs/include/optimizer/utils/convex_hull.hpp"

#include "external/libqhullcpp/RboxPoints.h"
#include "external/libqhullcpp/QhullError.h"
#include "external/libqhullcpp/QhullQh.h"
#include "external/libqhullcpp/QhullFacet.h"
#include "external/libqhullcpp/QhullFacetList.h"
#include "external/libqhullcpp/QhullLinkedList.h"
#include "external/libqhullcpp/QhullVertex.h"
#include "external/libqhullcpp/Qhull.h"
#include <boost/assert.hpp>
#include <string>
#include <sstream>

using orgQhull::Qhull;
using orgQhull::QhullError;
using orgQhull::QhullFacet;
using orgQhull::QhullFacetList;
using orgQhull::QhullQh;
using orgQhull::RboxPoints;
using orgQhull::QhullVertex;
using orgQhull::QhullVertexSet;

namespace Optimizer { namespace details {
    // Modified QuickHull algorithm using d-dimensional Beneath-Beyond
    // Reference: N. Perevoshchikova, et al., 2012, Computational Materials Science.
    // "A convex hull algorithm for a grid minimization of Gibbs energy as initial step 
    //    in equilibrium calculations in two-phase multicomponent alloys"
    void lower_convex_hull ( const std::vector<std::vector<double>> &points,
                             const std::vector<std::size_t> &dependent_dimensions ) {
        BOOST_ASSERT(points.size() > 0);
        const std::size_t point_dimension = points.begin()->size();
        const std::size_t point_count = points.size();
        const std::size_t point_buffer_size = point_dimension * point_count;
        double point_buffer[point_buffer_size-1];
        std::size_t buffer_offset = 0;
        std::string Qhullcommand = "Qt";
        // Copy all of the points into a buffer compatible with Qhull
        for (auto pt : points) {
            for (auto coord : pt) {
                if (buffer_offset >= point_buffer_size) break;
                point_buffer[buffer_offset++] = coord;
            }
        }
        BOOST_ASSERT(buffer_offset == point_buffer_size);
        
        // Mark dependent dimensions for Qhull so they can be discarded
        for (auto dim : dependent_dimensions) {
            std::stringstream stream;
            // Qhull command "Qbk:0Bk:0" drops dimension k from consideration
            stream << " " << "Qb" << dim << ":0B" << dim << ":0";
            Qhullcommand += stream.str();
        }
        std::cout << "DEBUG: Qhullcommand: " << Qhullcommand.c_str() << std::endl;
        Qhull qhull("", point_dimension, point_count, point_buffer, Qhullcommand.c_str());
        QhullFacetList facets = qhull.facetList();
        // TODO: Test: Print only the bottom-oriented facets
        for (auto facet : facets) {
            if (!facet.isTopOrient()) std::cout << facet;
        }
    }
} // namespace details
} // namespace Optimizer