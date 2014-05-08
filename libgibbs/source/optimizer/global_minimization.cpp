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
template <typename ValueType>
GlobalMinimizer<ValueType>::GlobalMinimizer (
    std::list<CompositionSet> const &phase_list,
    sublattice_set const &sublset,
    evalconditions const& conditions,
    GlobalMinimizationMethod method, 
    Potential pot) {
};
} // namespace Optimizer