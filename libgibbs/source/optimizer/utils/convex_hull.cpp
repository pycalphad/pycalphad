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
#include "external/libqhullcpp/QhullHyperplane.h"
#include "external/libqhullcpp/QhullLinkedList.h"
#include "external/libqhullcpp/QhullVertex.h"
#include "external/libqhullcpp/QhullVertexSet.h"
#include "external/libqhullcpp/Qhull.h"
#include <boost/assert.hpp>
#include <string>
#include <sstream>
#include <algorithm>
#include <functional>

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
                             const std::vector<std::size_t> &dependent_dimensions,
                             const double critical_edge_length
                           ) {
        BOOST_ASSERT(points.size() > 0);
        BOOST_ASSERT(critical_edge_length > 0);
        const std::size_t point_dimension = points.begin()->size();
        const std::size_t point_count = points.size();
        const std::size_t point_buffer_size = point_dimension * point_count;
        std::vector<std::vector<double>> candidate_points; // vertices of tie hyperplanes
        double point_buffer[point_buffer_size-1];
        std::size_t buffer_offset = 0;
        std::string Qhullcommand = "Qt"; // triangulate convex hull (make all facets simplicial)
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
        // Make the call to Qhull
        Qhull qhull("", point_dimension, point_count, point_buffer, Qhullcommand.c_str());
        // Get all of the facets
        QhullFacetList facets = qhull.facetList();
        for (auto facet : facets) {
            if (facet.isDefined() && facet.isGood()) {
                double orientation = *(facet.hyperplane().constEnd()-1); // last coordinate (energy)
                if (orientation > 0) continue; // consider only the facets of the lower convex hull
                QhullVertexSet vertices = facet.vertices();
                const std::size_t vertex_count = vertices.size();
                // Only facets with edges beyond the critical length are candidate tie hyperplanes
                // Check the length of all edges (dimension 1) in the facet
                for (auto vertex1 = 0; vertex1 < vertex_count; ++vertex1) {
                    std::vector<double> pt_vert1 = vertices[vertex1].point().toStdVector();
                    pt_vert1.pop_back(); // Remove the last coordinate (energy) for this check
                    for (auto vertex2 = 0; vertex2 < vertex1; ++vertex2) {
                        std::vector<double> pt_vert2 = vertices[vertex2].point().toStdVector();
                        pt_vert2.pop_back(); // Remove the last coordinate (energy) for this check
                        std::vector<double> difference ( pt_vert2.size() );
                        double distance = 0;
                        // Subtract vertex1 from vertex2 to get the distance
                        std::transform (pt_vert2.begin(), pt_vert2.end(), 
                                        pt_vert1.begin(), difference.begin(), std::minus<double>());
                        // Sum the square of all elements of vertex2-vertex1
                        for (auto coord : difference) distance += std::pow(coord,2);
                        // Square root the result
                        distance = sqrt(distance);
                        // if the edge length is large enough, this is a candidate tie hyperplane
                        if (distance > critical_edge_length) {
                          std::cout << "Edge length: " << distance << std::endl;
                          std::cout << "Vertex1: ";
                          for (auto coord : pt_vert1) std::cout << coord << ",";
                          std::cout << std::endl;
                          std::cout << "Vertex2: ";
                          for (auto coord : pt_vert2) std::cout << coord << ",";
                          std::cout << std::endl;
                          candidate_points.push_back(pt_vert1);
                          candidate_points.push_back(pt_vert2);
                          std::cout << facet;
                        }
                    }
                }
            }
        }
        if (candidate_points.size() > 0) {
            // There is at least one tie hyperplane
            // First, remove duplicate points
            // too_similar is a binary predicate for determining if the minima are too close in state space
            auto too_similar = [] ( const std::vector<double> &a, const std::vector<double> &b ) {
                if ( a.size() != b.size() ) {
                    return false;
                }
                for ( auto i = 0; i < a.size(); ++i ) {
                    if ( fabs ( a[i]-b[i] ) > 0.1 ) {
                        return false;    // at least one element is different enough
                    }
                }
                return true; // all elements compared closely
            };
            std::sort ( candidate_points.begin(), candidate_points.end() );
            std::unique ( candidate_points.begin(), candidate_points.end(), too_similar );
            // Second, restore the dependent variables to the correct coordinate placement
            for (auto dim : dependent_dimensions) {
            }
        }
        else {
            // No tie hyperplanes have been found
        }
    }
} // namespace details
} // namespace Optimizer