/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// libgibbs_pch.hpp -- list of headers to precompile for libGibbs

#ifndef BOOST_SPIRIT_USE_PHOENIX_V3
#define BOOST_SPIRIT_USE_PHOENIX_V3 1
#endif
#include "libtdb/include/database.hpp"
#include "libtdb/include/structure.hpp"
#include "libgibbs/include/constraint.hpp"
#include "external/coin/IpIpoptApplication.hpp"
#include "external/coin/IpSolveStatistics.hpp"
#include <boost/spirit/include/support_utree.hpp>
