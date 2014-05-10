/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Calculate convex hull using Qhull / libqhullcpp

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/utils/convex_hull.hpp"
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
    std::vector<std::vector<double>> internal_lower_convex_hull (
                             const std::vector<std::vector<double>> &points,
                             const std::set<std::size_t> &dependent_dimensions,
                             const double critical_edge_length,
                             std::function<double(const std::vector<double>&)> calculate_objective
                           ) {
        BOOST_ASSERT(points.size() > 0);
        BOOST_ASSERT(critical_edge_length > 0);
        const double coplanarity_allowance = 0.001; // max energy difference (%/100) to still be on tie plane
        const std::size_t point_dimension = points.begin()->size();
        const std::size_t point_count = points.size();
        std::vector<std::vector<double>> candidate_points, final_points; // vertices of tie hyperplanes
        RboxPoints point_buffer;
        point_buffer.setDimension ( point_dimension );
        point_buffer.reserveCoordinates ( point_count );
        std::string Qhullcommand = "";
        if (points.size() == 1) { // Special case: No composition dependence
            auto return_point = restore_dependent_dimensions ( points.front(), dependent_dimensions );
            final_points.emplace_back ( std::move ( return_point ) );
            return final_points;
        }
        // Copy all of the points into a buffer compatible with Qhull
        for (auto pt : points) {
            point_buffer.append ( QhullPoint ( point_dimension, &pt[0] ) );
        }
        
        // Mark dependent dimensions for Qhull so they can be discarded
        for (auto dim : dependent_dimensions) {
            std::stringstream stream;
            // Qhull command "Qbk:0Bk:0" drops dimension k from consideration
            stream << " " << "Qb" << dim << ":0B" << dim << ":0";
            Qhullcommand += stream.str();
        }
        std::cout << "DEBUG: Qhullcommand: " << Qhullcommand.c_str() << std::endl;
        // Make the call to Qhull
        Qhull qhull ( point_buffer, Qhullcommand.c_str() );
        // Get all of the facets
        QhullFacetList facets = qhull.facetList();
        for (auto facet : facets) {
            if (facet.isDefined() && facet.isGood() /*&& facet.isSimplicial()*/) {
                double orientation = *(facet.hyperplane().constEnd()-1); // last coordinate (energy)
                if (orientation > 0) continue; // consider only the facets of the lower convex hull
                QhullVertexSet vertices = facet.vertices();
                const std::size_t vertex_count = vertices.size();

                // Only facets with edges beyond the critical length are candidate tie hyperplanes
                // Check the length of all edges (dimension 1) in the facet
                for (auto vertex1 = 0; vertex1 < vertex_count; ++vertex1) {
                    std::vector<double> pt_vert1 = vertices[vertex1].point().toStdVector();
                    const double vertex1_energy = pt_vert1.back();
                    pt_vert1.pop_back(); // Remove the last coordinate (energy) for this check
                    for (auto vertex2 = 0; vertex2 < vertex1; ++vertex2) {
                        std::vector<double> pt_vert2 = vertices[vertex2].point().toStdVector();
                        const double vertex2_energy = pt_vert2.back();
                        pt_vert2.pop_back(); // Remove the last coordinate (energy) for this check
                        std::vector<double> difference ( pt_vert2.size() );
                        std::vector<double> midpoint ( pt_vert2.size() ); // midpoint of the edge
                        std::transform (pt_vert2.begin(), pt_vert2.end(), 
                                        pt_vert1.begin(), midpoint.begin(), std::plus<double>() );
                        for (auto &coord : midpoint) coord /= 2;
                        const double lever_rule_energy = (vertex1_energy + vertex2_energy)/2;
                        midpoint.pop_back(); // remove energy coordinate
                        midpoint = restore_dependent_dimensions ( midpoint, dependent_dimensions );
                        const double true_energy = calculate_objective ( midpoint );
                        // If the true energy is "much" greater, it's a true tie line
                        std::cout << "pt_vert1(" << vertices[vertex1].point().id() << "): ";
                        for (auto &coord : pt_vert1) std::cout << coord << ",";
                        std::cout << ":: ";
                        std::cout << "pt_vert2(" << vertices[vertex2].point().id() << "): ";
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
                        std::transform (pt_vert2.begin(), pt_vert2.end(), 
                                        pt_vert1.begin(), difference.begin(), std::minus<double>() );
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
                          candidate_points.push_back(restore_dependent_dimensions (pt_vert1, dependent_dimensions ));
                          candidate_points.push_back(restore_dependent_dimensions (pt_vert2, dependent_dimensions ));
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
                    if ( fabs ( a[i]-b[i] ) > 0.01 ) {
                        return false;    // at least one element is different enough
                    }
                }
                return true; // all elements compared closely
            };
            std::sort ( candidate_points.begin(), candidate_points.end() );
            auto new_end = std::unique ( candidate_points.begin(), candidate_points.end(), too_similar );
            // Fix the deduplicated point list to have no empty elements
            candidate_points.resize( std::distance( candidate_points.begin(), new_end ) );
            std::cout << "CANDIDATE POINTS AFTER DEDUPLICATION" << std::endl;
            for (auto pt : candidate_points) {
                for (auto coord : pt) {
                    std::cout << coord << ",";
                }
                std::cout << std::endl;
            }
            std::cout << "candidate_points.size() = " << candidate_points.size() << std::endl;
            // Second, restore the dependent variables to the correct coordinate placement
            for (const auto pt : candidate_points) {
                final_points.emplace_back ( restore_dependent_dimensions ( pt, dependent_dimensions ) );
            }
        }
        else {
            // No tie hyperplanes have been found
            // Return the point with the lowest energy (last coordinate)
            auto minimum_point_iterator = points.cbegin();
            for (auto pt = points.cbegin(); pt != points.cend(); ++pt) {
                // Check the energy values
                if (*(minimum_point_iterator->cend()-1) > *(pt->cend()-1)) {
                    // This point is lower in energy
                    minimum_point_iterator = pt;
                }
            }
            std::vector<double> return_point ( *minimum_point_iterator );
            return_point.pop_back(); // Remove energy coordinate
            final_points.emplace_back ( std::move ( return_point ) );
        }
        std::cout << "FINAL TIE POINTS" << std::endl;
        for (auto pt : final_points) {
            for (auto coord : pt) {
                std::cout << coord << ",";
            }
            std::cout << std::endl;
        }
        return final_points;
    }
    // Add the dependent site fraction coordinates back to the point
    std::vector<double> restore_dependent_dimensions (
        const std::vector<double> &point, 
        const std::set<std::size_t> &dependent_dimensions) {
        std::vector<double> final_point;
        final_point.reserve ( point.size() + dependent_dimensions.size() );
        std::size_t sublattice_offset = 0;
        auto iter = point.cbegin();
        for (auto dim : dependent_dimensions) {
            double point_sum = 0;
            for (auto coord = sublattice_offset; coord < dim; ++coord) {
                std::cout << "sublattice_offset: " << sublattice_offset << " coord: " << coord << " dim: " << dim << std::endl;
                point_sum += *iter;
                final_point.push_back ( *iter );
                if (iter != point.cend()) ++iter;
            }
            // add back the dependent component
            final_point.emplace_back ( 1 - point_sum ); // dependent coordinate is 1 - independents
            sublattice_offset = dim+1; // move to next sublattice
        }
        return final_point;
    }
} // namespace details
} // namespace Optimizer