/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Classes and routines for handling convex hulls with internal degrees of freedom

#ifndef INCLUDED_HULL_MAPPING
#define INCLUDED_HULL_MAPPING

#include <map>
#include <set>
#include <vector>

namespace Optimizer { namespace details {
/* The purpose of ConvexHullMap is to keep track of all of the internal convex
 * hulls of all of the phases during global minimization. The points of each phase's
 * internal hull are mapped to the global composition space. Then a global hull is
 * calculated and also stored, but this time it's stored as facets rather than points.
 * We apply composition and activity constraints to the facets to locate the candidate.
 * The point IDs of the vertices of the candidate facet are used to find each vertex's
 * internal degrees of freedom; this is the composition of that phase. Using the constraints
 * we fix a point inside the facet and use the lever rule to find the phase fractions.
 */
template <typename CoordinateType>
class ConvexHullMap {
public:
    typedef std::vector<CoordinateType> PointType;
    typedef std::vector<const PointType> PointContainerType;
    typename PointType::const_iterator find_internal_point_from_global_id ( const std::size_t index ) const;
    void reserve_points ( const std::size_t point_count ) {
        hull_phase_internal_points.reserve ( point_count );
        hull_global_points.reserve ( point_count );
    };
    void add_point ( const PointType &internal_coordinates, const PointType &global_coordinates );
private:
    // These two must be manually kept in sync using point IDs!
    PointContainerType hull_phase_internal_points;
    PointContainerType hull_global_points;
    // using set::lower_bound(), can get iterator to start/end of phase internal points
    // the difference between the actual value of the key and the queried value gives the offset
    std::set<std::size_t,typename PointType::const_iterator> phase_index_bounds;
};

} // namespace details
} // namespace Optimizer

#endif