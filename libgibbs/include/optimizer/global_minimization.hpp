/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Declarations for global minimization of a thermodynamic potential

#ifndef INCLUDED_GLOBAL_MINIMIZATION
#define INCLUDED_GLOBAL_MINIMIZATION

#include "libgibbs/include/compositionset.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/utils/hull_mapping.hpp"
#include "libgibbs/include/utils/for_each_pair.hpp"
#include "libgibbs/include/utils/site_fraction_convert.hpp"
#include "libtdb/include/logging.hpp"
#include <boost/assert.hpp>
#include <boost/noncopyable.hpp>
#include <functional>
#include <list>
#include <limits>
#include <set>

namespace Optimizer {

/* GlobalMinimizer performs global minimization of the specified
 * thermodynamic potential. Energy manifolds are calculated for
 * all phases in the global composition space and each phase's
 * internal degrees of freedom. Constraints can be added incrementally
 * to identify the equilibrium tie hyperplane and fix a position in it.
 */
template <
typename FacetType,
typename CoordinateType = double, 
typename EnergyType = CoordinateType
>
class GlobalMinimizer : private boost::noncopyable {
public:
    typedef details::ConvexHullMap<CoordinateType,EnergyType> HullMapType;
private:
    HullMapType hull_map;
    std::vector<FacetType> candidate_facets;
    mutable logger class_log;
public:
    /*
     * The type definitions here get a bit intense. We do this so we keep our
     * global minimization code flexible and extensible. By farming out pieces of the
     * process to functors, we retain the ability to swap out and try different sampling
     * methods, minimization methods, etc., without modifying the class definition.
     */
    typedef typename HullMapType::PointType PointType;
    typedef typename HullMapType::GlobalPointType GlobalPointType;
    typedef std::function<EnergyType(PointType const&)> CalculateEnergyFunctor;
    typedef std::function<double(const std::size_t, const std::size_t)>  CalculateGlobalEnergyFunctor;
    typedef std::function<std::vector<PointType>(CompositionSet const&, 
                                                 sublattice_set const&,
                                                 evalconditions const&)> PointSampleFunctor;
    typedef std::function<std::vector<PointType>(std::vector<PointType> const&,
                                                 std::set<std::size_t> const&,
                                                 CalculateEnergyFunctor const&)> InternalHullFunctor;
    typedef std::function<std::vector<FacetType>(std::vector<PointType> const&,
                                                CalculateGlobalEnergyFunctor const&)> GlobalHullFunctor;

    /* GlobalMinimizer works by taking the phase information for the system and a
     * list of functors that implement point sampling and convex hull calculation.
     * Once GlobalMinimizer is constructed, the user can filter against the calculated grid.
     */
    GlobalMinimizer ( 
            std::map<std::string,CompositionSet> const &phase_list,
            sublattice_set const &sublset,
            evalconditions const& conditions,
            PointSampleFunctor &sample_points,
            InternalHullFunctor &phase_internal_hull,
            GlobalHullFunctor &global_hull
                    ) {
        BOOST_LOG_NAMED_SCOPE ( "GlobalMinimizer::GlobalMinimizer" );
        BOOST_LOG_CHANNEL_SEV ( class_log, "optimizer", debug ) << "enter ctor";
        std::vector<PointType> temporary_hull_storage;

        for ( auto comp_set = phase_list.begin(); comp_set != phase_list.end(); ++comp_set ) {
            std::set<std::size_t> dependent_dimensions;
            std::size_t current_dependent_dimension = 0;
            
            // Determine the indices of the dependent dimensions
            boost::multi_index::index<sublattice_set,phase_subl>::type::iterator ic0,ic1;
            int sublindex = 0;
            ic0 = boost::multi_index::get<phase_subl> ( sublset ).lower_bound ( boost::make_tuple ( comp_set->first, sublindex ) );
            ic1 = boost::multi_index::get<phase_subl> ( sublset ).upper_bound ( boost::make_tuple ( comp_set->first, sublindex ) );
            
            while ( ic0 != ic1 ) {
                const std::size_t number_of_species = std::distance ( ic0,ic1 );
                if ( number_of_species > 0 ) {
                    // Last component is dependent dimension
                    current_dependent_dimension += (number_of_species-1);
                    dependent_dimensions.insert(current_dependent_dimension);
                    ++current_dependent_dimension;
                }
                // Next sublattice
                ++sublindex;
                ic0 = boost::multi_index::get<phase_subl> ( sublset ).lower_bound ( boost::make_tuple ( comp_set->first, sublindex ) );
                ic1 = boost::multi_index::get<phase_subl> ( sublset ).upper_bound ( boost::make_tuple ( comp_set->first, sublindex ) );
            }
            // Sample the composition space of this phase
            auto phase_points = sample_points ( comp_set->second, sublset, conditions );
            // Create a callback function for energy calculation for this phase
            auto calculate_energy = [comp_set,&conditions] (const PointType& point) {
                return comp_set->second.evaluate_objective(conditions,comp_set->second.get_variable_map(),const_cast<EnergyType*>(&point[0]));
            };
            // Calculate the phase's internal convex hull and store the result
            auto phase_hull_points = phase_internal_hull ( phase_points, dependent_dimensions, calculate_energy );
            // TODO: Apply phase-specific constraints to internal dof and globally
            // Add all points from this phase's convex hull to our internal hull map
            for ( auto point : phase_hull_points ) {
                // All points added to the hull_map could possibly be on the global hull
                auto global_point = convert_site_fractions_to_mole_fractions ( comp_set->first, sublset, point );
                PointType ordered_global_point;  
                ordered_global_point.reserve ( global_point.size()+1 );
                for ( auto pt : global_point ) ordered_global_point.push_back ( pt.second );
                double energy = calculate_energy ( point );
                hull_map.insert_point ( 
                                       comp_set->first, 
                                       energy, 
                                       point,
                                       global_point
                                      );
                ordered_global_point.push_back ( energy );
                temporary_hull_storage.push_back ( std::move ( ordered_global_point ) );
            }
        }
        // Calculate the "true energy" of the midpoint of two points, based on their IDs
        // If the phases are distinct, the "true energy" is infinite (indicates true line)
        auto calculate_global_midpoint_energy = [this,&conditions,&phase_list] 
            (const std::size_t point1_id, const std::size_t point2_id) 
            { 
                BOOST_ASSERT ( point1_id < hull_map.size() );
                BOOST_ASSERT ( point2_id < hull_map.size() );
                if ( point1_id == point2_id) return hull_map[point1_id].energy;
                if (hull_map[point1_id].phase_name != hull_map[point2_id].phase_name) {
                    // Can't calculate a "true energy" if the tie points are different phases
                    return std::numeric_limits<EnergyType>::max();
                }
                // Return the energy of the average of the internal degrees of freedom
                else {
                    PointType midpoint ( hull_map[point1_id].internal_coordinates );
                    PointType point2 ( hull_map[point2_id].internal_coordinates );
                    auto current_comp_set = phase_list.find ( hull_map[point1_id].phase_name );
                    std::transform ( 
                                    point2.begin(), 
                                    point2.end(),
                                    midpoint.begin(),
                                    midpoint.begin(),
                                    std::plus<EnergyType>()
                                ); // sum points together
                    for (auto &coord : midpoint) coord /= 2; // divide by two
                    auto calculate_energy = [&] (const PointType& point) {
                        return current_comp_set->second.evaluate_objective(conditions,current_comp_set->second.get_variable_map(),const_cast<EnergyType*>(&point[0]));
                    };
                    return  calculate_energy ( midpoint ); 
            }
        };
        // TODO: Add points and set options related to activity constraints here
        // Determine the facets on the global convex hull of all phase's energy landscapes
        candidate_facets = global_hull ( temporary_hull_storage, calculate_global_midpoint_energy );
        BOOST_LOG_SEV ( class_log, debug ) << "candidate_facets.size() = " << candidate_facets.size();
    }
    
    std::vector<typename HullMapType::HullEntryType> find_tie_points ( 
                       evalconditions const& conditions,
                       const std::size_t critical_edge_length
                                                                     ) {
        BOOST_LOG_NAMED_SCOPE ( "GlobalMinimizer::find_tie_points" );
        // Filter candidate facets based on user-specified constraints
        std::set<std::size_t> candidate_ids; // ensures returned points are unique
        std::vector<typename HullMapType::HullEntryType> candidates;
        BOOST_LOG_SEV ( class_log, debug ) << "candidate_facets.size() = " << candidate_facets.size();
        for ( auto facet : candidate_facets ) {
            BOOST_LOG_SEV ( class_log, debug ) << "facet.vertices.size() = " << facet.vertices.size();
            bool failed_conditions = false;
            for ( auto species : conditions.xfrac ) {
            }
            boost::numeric::ublas::vector<CoordinateType> trial_point ( conditions.xfrac.size()+1 );
            for ( auto coord = conditions.xfrac.begin(); coord !=  conditions.xfrac.end(); ++coord) {
                trial_point [ std::distance(conditions.xfrac.begin(),coord) ] = coord->second;
            }
            trial_point [ conditions.xfrac.size() ] = 1;
            auto trial_vector = boost::numeric::ublas::prod ( facet.basis_matrix, trial_point );
            for ( auto coord : trial_vector ) {
                if ( coord < 0 ) {
                    failed_conditions = true;
                    break;
                }
            }
                /*
                double min_extent = 1, max_extent = 0; // extents of this coordinate
                for ( auto point = facet.vertices.begin(); point != facet.vertices.end(); ++point ) {
                    const std::size_t point_id = *point;
                    auto point_entry = hull_map [ point_id ];
                    auto point_coordinate = point_entry.global_coordinates [ species.first ];
                    BOOST_LOG_SEV ( class_log, debug ) << "point_coordinate = " << point_coordinate;
                    min_extent = std::min ( min_extent, point_coordinate );
                    max_extent = std::max ( max_extent, point_coordinate );
                }
                BOOST_LOG_SEV ( class_log, debug ) << "min_extent = " << min_extent;
                BOOST_LOG_SEV ( class_log, debug ) << "max_extent = " << max_extent;
                if ( species.second >= min_extent && species.second <= max_extent ) {
                    BOOST_LOG_SEV ( class_log, debug ) << "conditions passed";
                }
                else {
                    BOOST_LOG_SEV ( class_log, debug ) << "conditions failed"; 
                    failed_conditions = true; 
                    break; 
                }
                */
            if (!failed_conditions) {
                std::stringstream logbuf;
                logbuf << "Candidate facet ";
                for ( auto point : facet.vertices ) {
                    const std::size_t point_id = point;
                    auto point_entry = hull_map [ point_id ];
                    logbuf << "[";
                    for ( auto coord : point_entry.internal_coordinates ) {
                        logbuf << coord << ",";
                    }
                    logbuf << "]";
                    logbuf << "{";
                    for ( auto coord : point_entry.global_coordinates ) {
                        logbuf << coord.first << ":" << coord.second << ",";
                    }
                    logbuf << "}";
                }
                BOOST_LOG_SEV ( class_log, debug ) << logbuf.str();
                // this facet satisfies all the conditions; return it
                for_each_pair (facet.vertices.begin(), facet.vertices.end(), 
                    [this,&candidate_ids,critical_edge_length](
                        decltype( facet.vertices.begin() ) point1, 
                       decltype( facet.vertices.begin() ) point2
                    ) { 
                        const std::size_t point1_id = *point1;
                        const std::size_t point2_id = *point2;
                        auto point1_entry = hull_map [ point1_id ];
                        auto point2_entry = hull_map [ point2_id ];
                        if ( point1_entry.phase_name != point2_entry.phase_name ) {
                            // phases differ; definitely a tie line
                            candidate_ids.insert ( point1_id );
                            candidate_ids.insert ( point2_id );
                        }
                        else {
                            // phases are the same -- does a tie line span a miscibility gap?
                            // use internal coordinates to check
                            CoordinateType distance = 0;
                            auto difference = point2_entry.internal_coordinates ;
                            auto diff_iter = difference.begin();
                            for ( auto coord : point1_entry.internal_coordinates ) {
                                *(diff_iter++) -= coord;
                            }
                            for (auto coord : difference) distance += std::pow(coord,2);
                            if (distance > critical_edge_length) {
                                // the tie line is sufficiently long
                                candidate_ids.insert ( point1_id );
                                candidate_ids.insert ( point2_id );
                            }
                            // Not a tie line; just add one point to ensure the phase appears
                            else {
                                candidate_ids.insert ( point1_id );
                            }
                        }
                    });
            }
        }
        for (auto point_id : candidate_ids) {
            candidates.push_back ( hull_map [ point_id ] );
        }
        return std::move( candidates );
    }
};

} //namespace Optimizer


#endif