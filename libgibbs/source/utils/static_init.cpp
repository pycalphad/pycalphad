/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/utils/enum_handling.hpp"
#include "libgibbs/include/conditions.hpp"

// static member initialization for template classes


template<> char const* enumStrings<Optimizer::PhaseStatus>::data[] = {"ENTERED", "DORMANT", "FIXED", "SUSPENDED"};
