/*=============================================================================
	Copyright (c) 2012-2014 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// declaration for functions related to EZD global minimization
// Reference: Maria Emelianenko, Zi-Kui Liu, and Qiang Du.
// "A new algorithm for the automation of phase diagram calculation." Computational Materials Science 35.1 (2006): 61-74.

#ifndef INCLUDED_EZD_MINIMIZATION
#define INCLUDED_EZD_MINIMIZATION

#include "libgibbs/include/models.hpp"
#include "libgibbs/include/compositionset.hpp"
#include "libgibbs/include/constraint.hpp"
#include "libgibbs/include/conditions.hpp"
#include <boost/bimap.hpp>
#include <vector>

namespace Optimizer {

std::vector<std::map<std::string,double>>  LocateMinima(
		CompositionSet const &phase,
		sublattice_set const &sublset,
		evalconditions const& conditions,
		const std::size_t depth = 1
		);
}

#endif
