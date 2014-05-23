/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// declaration for PhaseStatus

#ifndef PHASESTATUS_INCLUDED
#define PHASESTATUS_INCLUDED

namespace Optimizer {
enum class PhaseStatus : unsigned int {
	ENTERED = 0, DORMANT = 1, FIXED = 2, SUSPENDED = 3
};
}

#endif
