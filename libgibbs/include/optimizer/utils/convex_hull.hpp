/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

#ifndef INCLUDED_CONVEX_HULL
#define INCLUDED_CONVEX_HULL

#include "external/libqhullcpp/QhullFacet.h"
#include <map>
#include <string>
#include <vector>
#include <set>
#include <functional>

namespace Optimizer {
    namespace details {
        // Calculation of the internal lower convex hull of a set of points
        std::vector<std::vector<double>> internal_lower_convex_hull ( 
        const std::vector<std::vector<double>> &points, 
        const std::set<std::size_t> &dependent_dimensions,
        const double critical_edge_length,
        const std::function<double(const std::vector<double>&)> calculate_objective
        );
        
        // Calculation of the global convex hull of a system
        std::vector<orgQhull::QhullFacet> global_lower_convex_hull (
            const std::vector<std::vector<double>> &points,
            const double critical_edge_length,
            const std::function<double(const std::size_t, const std::size_t)> calculate_midpoint_energy
        );
    }
}

#endif