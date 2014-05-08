/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Declarations for global minimization of a thermodynamic potential

#ifndef INCLUDED_GLOBAL_MINIMIZATION
#define INCLUDED_GLOBAL_MINIMIZATION

#include "libgibbs/include/optimizer/utils/hull_mapping.hpp"
#include <boost/noncopyable.hpp>
#include <functional>
#include <list>


namespace Optimizer {

// Relevant forward declarations
class CompositionSet;
class sublattice_set;
class evalconditions;

/* GlobalMinimizer performs global minimization of the specified
 * thermodynamic potential. Energy manifolds are calculated for
 * all phases in the global composition space and each phase's
 * internal degrees of freedom. Constraints can be added incrementally
 * to identify the equilibrium tie hyperplane and fix a position in it.
 */
template <typename CoordinateType = double, typename EnergyType = CoordinateType>
class GlobalMinimizer : private boost::noncopyable {
public:
    GlobalMinimizer ( 
            std::map<std::string,CompositionSet> const &phase_list,
            sublattice_set const &sublset,
            evalconditions const& conditions,
            std::function<void()> &phase_internal_hull);
private:
    details::ConvexHullMap<CoordinateType,EnergyType> hull_map;
};

} //namespace Optimizer


#endif