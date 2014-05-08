/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// implementation of global minimization of thermodynamic potentials
// Various methods could be supported

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/compositionset.hpp"
#include "libgibbs/include/conditions.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/global_minimization.hpp"

namespace Optimizer {
template <typename CoordinateType, typename EnergyType>
GlobalMinimizer<CoordinateType,EnergyType>::GlobalMinimizer ( 
    std::map<std::string,CompositionSet> const &phase_list,
    sublattice_set const &sublset,
    evalconditions const& conditions,
    std::function<void()> &phase_internal_hull) {
}
} // namespace Optimizer