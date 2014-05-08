/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// implementation of convex hull mapping between internal/global composition space

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/utils/hull_mapping.hpp"

namespace Optimizer { namespace details {
/*
 * Add a point from a phase's internal convex hull to the global point map.
 */
template <typename CoordinateType, typename EnergyType>
void ConvexHullMap<CoordinateType,EnergyType>::insert_point ( 
    const std::string &phase_name,
    const EnergyType &energy, 
    const PointType &internal_coordinates,
    const GlobalPointType &global_coordinates
) {
    HullEntryType hull_entry;
    hull_entry.energy = energy;
    hull_entry.internal_coordinates = internal_coordinates;
    hull_entry.global_coordinates = global_coordinates;
    all_points.emplace_back ( std::move ( hull_entry ) );
}

/* 
 * Locate the entry corresponding to this global point ID.
 */
template <typename CoordinateType, typename EnergyType>
auto ConvexHullMap<CoordinateType,EnergyType>::find_entry_from_global_id ( const std::size_t index ) 
-> typename HullEntryContainerType::const_iterator const {
     auto entry_iterator = all_points.cbegin();
     std::advance ( entry_iterator, index );
     return entry_iterator;
 }
} // namespace details
} // namespace Optimizer