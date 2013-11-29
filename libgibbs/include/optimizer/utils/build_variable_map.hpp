/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#ifndef INCLUDED_BUILD_VARIABLE_MAP
#define INCLUDED_BUILD_VARIABLE_MAP

#include "libgibbs/include/models.hpp"
#include "libgibbs/include/conditions.hpp"
#include <boost/bimap.hpp>

// bridge function between Database and EnergyModels
sublattice_set build_variable_map(
		const Phase_Collection::const_iterator p_begin,
		const Phase_Collection::const_iterator p_end,
		const evalconditions &conditions,
		boost::bimap<std::string, int> &indices
		);

#endif
