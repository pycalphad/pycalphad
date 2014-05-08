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

namespace Optimizer {
enum class GlobalMinimizationMethod {
    Modified_EZD // Modified version of method of Emelianenko, Liu, and Du. Computational Materials Science 35.1 (2006): 61-74
};
enum class Potential {
    Gibbs_Energy // natural variables of T,P,N_i
};

/* GlobalMinimizer performs global minimization of the specified
 * thermodynamic potential. Energy manifolds are calculated for
 * all phases in the global composition space and each phase's
 * internal degrees of freedom. Constraints can be added incrementally
 * to identify the equilibrium tie hyperplane and fix a position in it.
 */
template <typename ValueType>
class GlobalMinimizer : private boost::noncopyable {
public:
    GlobalMinimizer (GlobalMinimizationMethod method = GlobalMinimizationMethod::Modified_EZD, 
                     Potential pot = Potential::Gibbs_Energy);
private:
    details::ConvexHullMap<ValueType> hullmap;
};

} //namespace Optimizer


#endif