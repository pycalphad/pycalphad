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
#include "libtdb/include/logging.hpp"
#include <boost/noncopyable.hpp>
#include <functional>
#include <list>

namespace Optimizer {

/* GlobalMinimizer performs global minimization of the specified
 * thermodynamic potential. Energy manifolds are calculated for
 * all phases in the global composition space and each phase's
 * internal degrees of freedom. Constraints can be added incrementally
 * to identify the equilibrium tie hyperplane and fix a position in it.
 */
template <typename CoordinateType = double, typename EnergyType = CoordinateType>
class GlobalMinimizer : private boost::noncopyable {
public:
    template <typename Functor> GlobalMinimizer ( 
            std::map<std::string,CompositionSet> &phase_list,
            sublattice_set const &sublset,
            evalconditions const& conditions,
            Functor &phase_internal_hull) {
        BOOST_LOG_NAMED_SCOPE ( "GlobalMinimizer::GlobalMinimizer" );
        BOOST_LOG_CHANNEL_SEV ( class_log, "optimizer", debug ) << "enter ctor";
        //BOOST_LOG_SEV ( opto_log, debug ) << minima.size() << " minima detected from global minimization";
        
        for (auto comp_set = phase_list.begin(); comp_set != phase_list.end();) {
            ++comp_set;
        }
        /*if ( minima.size() == 0 ) {
         * // We didn't find any minima!
         * // This could indicate a problem, or just that we didn't look hard enough
         * BOOST_LOG_SEV ( opto_log, critical ) << "Global minimization found no energy minima";
         * const std::string compset_name ( main_compset.name() );
         * // TODO: What about the starting point?!
         * comp_sets.emplace ( compset_name, std::move ( main_compset ) );
            }
            
            if ( minima.size() == 1 ) {
                // No miscibility gap, no need to create new composition sets
                // Use the minimum found during global minimization as the starting point
                main_compset.set_starting_point ( * ( minima.begin() ) );
                const std::string compset_name ( main_compset.name() );
                comp_sets.emplace ( compset_name, std::move ( main_compset ) );
            }
            if ( minima.size() > 1 ) {
                // Miscibility gap detected; create a new composition set for each minimum
                std::size_t compsetcount = 1;
                std::map<std::string, CompositionSet>::iterator it;
                for ( auto min = minima.begin(); min != minima.end(); ++min ) {
                    if ( min != minima.begin() ) {
                        ++compsetcount;
                        ++activephases;
            }
            std::stringstream compsetname;
            compsetname << main_compset.name() << "#" << compsetcount;
            
            // Set starting point
            const auto new_starting_point = ast_copy_with_renamed_phase ( *min, main_compset.name(), compsetname.str() );
            // Copy from PHASENAME to PHASENAME#N
            phase_col[compsetname.str()] = phase_col[main_compset.name()];
            conditions.phases[compsetname.str()] = conditions.phases[main_compset.name()];
            it = comp_sets.emplace ( compsetname.str(), CompositionSet ( main_compset, new_starting_point, compsetname.str() ) ).first;
            }
            // Remove PHASENAME
            // PHASENAME was renamed to PHASENAME#1
            BOOST_LOG_SEV ( opto_log, debug ) << "Removing old phase " << main_compset.name();
            auto remove_phase_iter = phase_col.find ( main_compset.name() );
            auto remove_conds_phase_iter = conditions.phases.find ( main_compset.name() );
            if ( remove_phase_iter != phase_col.end() ) {
                phase_col.erase ( remove_phase_iter );
            }
            if ( remove_conds_phase_iter != conditions.phases.end() ) {
                conditions.phases.erase ( remove_conds_phase_iter );
            }
            }
            }*/
    }
private:
    details::ConvexHullMap<CoordinateType,EnergyType> hull_map;
    mutable logger class_log;
};

} //namespace Optimizer


#endif