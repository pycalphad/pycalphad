/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// implementation of convex hull mapping between internal/global composition space

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/utils/hull_mapping.hpp"

namespace Optimizer { namespace details {
template <typename ValueType>
void ConvexHullMap<ValueType>::add_point ( 
        const ConvexHullMap<ValueType>::PointType &internal_coordinates, 
        const ConvexHullMap<ValueType>::PointType &global_coordinates ) {
    hull_phase_internal_points.insert_after ( hull_phase_internal_points.end(), internal_coordinates );
    hull_global_points.insert_after ( hull_global_points.end(), global_coordinates );
}
template <typename ValueType>
typename ConvexHullMap<ValueType>::PointContainerType::const_iterator 
 ConvexHullMap<ValueType>::find_internal_point_from_global_id ( const std::size_t index ) const {
     
 }
} // namespace details
} // namespace Optimizer