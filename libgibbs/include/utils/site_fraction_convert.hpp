/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Convert site fraction coordinates to mole fraction using the sublattice configuration
#ifndef INCLUDED_SITE_FRACTION_CONVERT
#define INCLUDED_SITE_FRACTION_CONVERT

#include "libgibbs/include/models.hpp"
#include <vector>

// Map internal phase coordinates to global composition space
template <typename CoordinateType>
std::vector<CoordinateType> convert_site_fractions_to_mole_fractions (
    const sublattice_set &sublset,
    const std::vector<CoordinateType> &internal_coordinates
                                                                     ) {
    std::vector<CoordinateType> global_coordinates;
    return global_coordinates;
};


#endif