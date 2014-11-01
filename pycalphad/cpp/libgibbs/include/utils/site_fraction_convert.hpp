/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Convert site fraction coordinates to mole fraction using the sublattice configuration
#ifndef INCLUDED_SITE_FRACTION_CONVERT
#define INCLUDED_SITE_FRACTION_CONVERT

#include "libgibbs/include/models.hpp"
#include <boost/assert.hpp>
#include <vector>
#include <map>
#include <string>

// Map internal phase coordinates to global composition space
// Input:  the name of the phase
// Input:  a sublattice_set containing the phase's configuration
// Input:  the full internal coordinates (including dependent coordinates)
//         Note that components of internal_coordinates must be ordered the same as sublset
// Output: the full global composition coordinates (including dependent coordinates) mapped by component name
template <typename CoordinateType>
std::map<std::string,CoordinateType> convert_site_fractions_to_mole_fractions (
    const std::string &phase_name,
    const sublattice_set &sublset,
    const std::vector<CoordinateType> &internal_coordinates) {
    
    // map component name to a value
    std::map<std::string,CoordinateType> component_mole_fraction_numerator;
    CoordinateType component_mole_fraction_denominator;
    std::map<std::string,CoordinateType> global_coordinates;
    // Get the first sublattice for this phase
    boost::multi_index::index<sublattice_set,phase_subl>::type::iterator subl_start,subl_end;
    int sublindex = 0;
    std::size_t internal_coordinate_index = 0;
    subl_start = boost::multi_index::get<phase_subl> ( sublset ).lower_bound ( boost::make_tuple ( phase_name, sublindex ) );
    subl_end = boost::multi_index::get<phase_subl> ( sublset ).upper_bound ( boost::make_tuple ( phase_name, sublindex ) );

    while ( subl_start != subl_end ) {
        for ( auto current_component = subl_start; current_component != subl_end; 
             ++current_component,++internal_coordinate_index ) {
            if ( current_component->species == "VA" ) continue; // vacancies don't contribute here
            
            BOOST_ASSERT ( internal_coordinate_index < internal_coordinates.size() );
            const CoordinateType site_fraction = internal_coordinates[ internal_coordinate_index ];
            
            // if we've already added to this component's numerator, we only increment it
            auto numerator_find = component_mole_fraction_numerator.find ( current_component->species );
            if ( numerator_find !=
                component_mole_fraction_numerator.end() ) {
                numerator_find->second += current_component->num_sites * site_fraction; // numerator
            }
            else {
                component_mole_fraction_numerator [ current_component->species ] = current_component->num_sites * site_fraction;
            }
            component_mole_fraction_denominator += current_component->num_sites * site_fraction;
        }

        // Next sublattice
        ++sublindex;
        subl_start = boost::multi_index::get<phase_subl> ( sublset ).lower_bound ( boost::make_tuple ( phase_name, sublindex ) );
        subl_end = boost::multi_index::get<phase_subl> ( sublset ).upper_bound ( boost::make_tuple ( phase_name, sublindex ) );
    }
    
    for ( auto component_numerator : component_mole_fraction_numerator ) {
        global_coordinates [ component_numerator.first ] = component_numerator.second / component_mole_fraction_denominator;
    }
    
    return global_coordinates;
};


#endif