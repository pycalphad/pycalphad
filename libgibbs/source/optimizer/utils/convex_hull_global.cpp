/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Calculate global convex hull using Qhull / libqhullcpp

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/utils/convex_hull.hpp"
#include "libgibbs/include/optimizer/utils/simplicial_facet.hpp"
#include "external/libqhullcpp/RboxPoints.h"
#include "external/libqhullcpp/QhullError.h"
#include "external/libqhullcpp/QhullQh.h"
#include "external/libqhullcpp/QhullFacet.h"
#include "external/libqhullcpp/QhullFacetList.h"
#include "external/libqhullcpp/QhullHyperplane.h"
#include "external/libqhullcpp/QhullLinkedList.h"
#include "external/libqhullcpp/QhullPoint.h"
#include "external/libqhullcpp/QhullVertex.h"
#include "external/libqhullcpp/QhullVertexSet.h"
#include "external/libqhullcpp/Qhull.h"
#include <boost/assert.hpp>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <algorithm>
#include <functional>
#include <cmath>

using orgQhull::RboxPoints;
using orgQhull::Qhull;
using orgQhull::QhullError;
using orgQhull::QhullFacet;
using orgQhull::QhullFacetList;
using orgQhull::QhullPoint;
using orgQhull::QhullQh;
using orgQhull::RboxPoints;
using orgQhull::QhullVertex;
using orgQhull::QhullVertexSet;

namespace Optimizer { namespace details {
// Modified QuickHull algorithm using d-dimensional Beneath-Beyond
// Reference: N. Perevoshchikova, et al., 2012, Computational Materials Science.
// "A convex hull algorithm for a grid minimization of Gibbs energy as initial step 
//    in equilibrium calculations in two-phase multicomponent alloys"
std::vector<SimplicialFacet<double>> global_lower_convex_hull (
    const std::vector<std::vector<double>> &points,
    const double critical_edge_length,
    const std::function<double(const std::size_t, const std::size_t)> calculate_midpoint_energy
) {
    BOOST_ASSERT(points.size() > 0);
    BOOST_ASSERT(critical_edge_length > 0);
    const double coplanarity_allowance = 0.001; // max energy difference (%/100) to still be on tie plane
    std::vector<SimplicialFacet<double>> candidates;
    const std::size_t point_dimension = points.begin()->size();
    const std::size_t point_count = points.size();
    std::set<std::size_t> candidate_point_ids; // vertices of tie hyperplanes
    RboxPoints point_buffer;
    point_buffer.setDimension ( point_dimension );
    point_buffer.reserveCoordinates ( point_count );
    std::string Qhullcommand = " ";
    
    // Copy all of the points into a buffer compatible with Qhull
    for (auto pt : points) {
        point_buffer.append ( QhullPoint ( point_dimension, &pt[0] ) );
    }
    
    // Mark last dimension as dependent for Qhull so it can be discarded
    std::stringstream stream;
    // Qhull command "Qbk:0Bk:0" drops dimension k from consideration
    // Remove dependent coordinate (second to last, energy should be last coordinate)
    stream << " " << "Qb" << point_dimension-2 << ":0B" << point_dimension-2 << ":0";
    Qhullcommand += stream.str();

    std::cout << "DEBUG: Qhullcommand: " << Qhullcommand.c_str() << std::endl;
    // Make the call to Qhull
    Qhull qhull ( point_buffer, Qhullcommand.c_str() );
    // Get all of the facets
    QhullFacetList facets = qhull.facetList();
  
    for (auto facet : facets) {
        bool already_added = false;
        if (facet.isDefined() && facet.isGood() && facet.isSimplicial() ) {
            double orientation = *(facet.hyperplane().constEnd()-1); // last coordinate (energy)
            if (orientation > 0) continue; // consider only the facets of the lower convex hull
            QhullVertexSet vertices = facet.vertices();
            const std::size_t vertex_count = vertices.size();
            
            // Only facets with edges beyond the critical length are candidate tie hyperplanes
            // Check the length of all edges (dimension 1) in the facet
            for (auto vertex1 = 0; vertex1 < vertex_count && !already_added; ++vertex1) {
                const std::size_t vertex1_point_id = vertices[vertex1].point().id();
                const double vertex1_energy = calculate_midpoint_energy ( vertex1_point_id, vertex1_point_id );
                std::cout << "vertex1_energy = " << vertex1_energy << std::endl;
                std::vector<double> pt_vert1 = vertices[vertex1].point().toStdVector();
                //pt_vert1.pop_back(); // Remove the last coordinate (energy) for this check
                for (auto vertex2 = 0; vertex2 < vertex1 && !already_added; ++vertex2) {
                    const std::size_t vertex2_point_id = vertices[vertex2].point().id();
                    const double vertex2_energy = calculate_midpoint_energy ( vertex2_point_id, vertex2_point_id );
                    std::cout << "vertex2_energy = " << vertex2_energy << std::endl;
                    std::vector<double> pt_vert2 = vertices[vertex2].point().toStdVector();
                    //pt_vert2.pop_back(); // Remove the last coordinate (energy) for this check
                    std::vector<double> difference ( pt_vert2.size() );
                    std::vector<double> midpoint ( pt_vert2.size() ); // midpoint of the edge
                    std::transform (pt_vert2.begin(), pt_vert2.end(), 
                                    pt_vert1.begin(), midpoint.begin(), std::plus<double>() );
                    for (auto &coord : midpoint) coord /= 2;
                    const double lever_rule_energy = (vertex1_energy + vertex2_energy)/2;
                    // This will return a type's max() if the phases are different (always true tie line)
                    const double true_energy = calculate_midpoint_energy
                                               ( 
                                                     vertex1_point_id, 
                                                     vertex2_point_id 
                                               );
                    // If the true energy is "much" greater, it's a true tie line
                    std::cout << "pt_vert1(" << vertex1_point_id << "): ";
                    for (auto &coord : pt_vert1) std::cout << coord << ",";
                    std::cout << ":: ";
                    std::cout << "pt_vert2(" << vertex2_point_id << "): ";
                    for (auto &coord : pt_vert2) std::cout << coord << ",";
                    std::cout << ":: ";
                    std::cout << "midpoint: ";
                    for (auto &coord : midpoint) std::cout << coord << ",";
                    std::cout << std::endl;
                    std::cout << "true_energy: " << true_energy << " lever_rule_energy: " << lever_rule_energy << std::endl;
                    // We use fabs() here so we don't accidentally flip the sign of the comparison
                    if ( (true_energy-lever_rule_energy)/fabs(lever_rule_energy) < coplanarity_allowance ) {
                        continue; // not a true tie line, skip it
                    }
                    
                    double distance = 0;
                    // Subtract vertex1 from vertex2 to get the distance
                    std::transform (pt_vert2.begin(), pt_vert2.end()-1, 
                                    pt_vert1.begin(), difference.begin(), std::minus<double>() );
                    // Sum the square of all elements of vertex2-vertex1
                    for (auto coord : difference) distance += std::pow(coord,2);
                    // Square root the result
                    distance = sqrt(distance);
                    std::cout << "Edge length: " << distance << std::endl;
                    std::cout << "Vertex1: ";
                    for (auto coord : pt_vert1) std::cout << coord << ",";
                    std::cout << std::endl;
                    std::cout << "Vertex2: ";
                    for (auto coord : pt_vert2) std::cout << coord << ",";
                    std::cout << std::endl;
                    SimplicialFacet<double> new_facet;
                    for ( auto vertex = vertices.begin(); vertex != vertices.end(); ++vertex ) {
                        new_facet.vertices.push_back ( vertex->point().id() );
                    }
                    for ( auto coord : facet.hyperplane() ) {
                        new_facet.normal.push_back ( coord );
                    }
                    new_facet.area = facet.facetArea( qhull.runId() );
                    candidates.push_back ( new_facet );
                    already_added = true;
                    std::cout << facet;
                }
            }
        }
    }
    return candidates;
}
} // namespace details
} // namespace Optimizer