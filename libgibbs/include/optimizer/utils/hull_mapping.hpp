/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Classes and routines for handling convex hulls with internal degrees of freedom

#ifndef INCLUDED_HULL_MAPPING
#define INCLUDED_HULL_MAPPING

#include <boost/noncopyable.hpp>
#include <map>
#include <string>
#include <vector>

namespace Optimizer { namespace details {

/* This class associates important metadata with each point on the convex hull
 * of a given phase. This way we always remember the internal degrees of freedom
 * that are contained within a phase.
 */
template <typename CoordinateType = double, typename EnergyType = CoordinateType> 
struct ConvexHullEntry {
    typedef std::vector<CoordinateType> PointType;
    typedef std::map<std::string,CoordinateType> GlobalPointType;
    std::string phase_name;
    CoordinateType energy;
    PointType internal_coordinates;
    GlobalPointType global_coordinates;
};
/* The purpose of ConvexHullMap is to keep track of all of the internal convex
 * hulls of all of the phases during global minimization. The points of each phase's
 * internal hull are mapped to the global composition space. Then a global hull is
 * calculated and also stored, but this time it's stored as facets rather than points.
 * We apply composition and activity constraints to the facets to locate the candidate.
 * The point IDs of the vertices of the candidate facet are used to find each vertex's
 * internal degrees of freedom; this is the composition of that phase. Using the constraints
 * we fix a point inside the facet and use the lever rule to find the phase fractions.
 */
template <typename CoordinateType = double, typename EnergyType = CoordinateType>
class ConvexHullMap : private boost::noncopyable {
public:
    typedef ConvexHullEntry<CoordinateType,EnergyType> HullEntryType;
    typedef std::vector<HullEntryType> HullEntryContainerType;
    typedef typename HullEntryType::PointType PointType;
    typedef typename HullEntryType::GlobalPointType GlobalPointType;
    const HullEntryType operator[] ( const std::size_t index ) { 
        auto return_iter = all_points.cbegin();
        std::advance(return_iter, index);
        return (const HullEntryType) *return_iter;
    };
    void insert_point ( const std::string &phase_name, const EnergyType &energy, 
                        const PointType &internal_coordinates, const GlobalPointType &global_coordinates ) {
        HullEntryType hull_entry;
        hull_entry.phase_name = phase_name;
        hull_entry.energy = energy;
        hull_entry.internal_coordinates = internal_coordinates;
        hull_entry.global_coordinates = global_coordinates;
        all_points.push_back ( hull_entry );
    };
private:
    // entries should be inserted in global ID order
    HullEntryContainerType all_points;
};


} // namespace details
} // namespace Optimizer

#endif