/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

#ifndef INCLUDED_CONVEX_HULL
#define INCLUDED_CONVEX_HULL

#include <vector>
#include <set>

namespace Optimizer {
    namespace details {
        // Calculation of the lower convex hull of a set of points
        std::vector<std::vector<double>> lower_convex_hull ( 
        const std::vector<std::vector<double>> &points, 
        const std::set<std::size_t> &dependent_dimensions,
        const double critical_edge_length
        );
    }
}

#endif