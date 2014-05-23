/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// warning_disable.hpp -- disabling some excessive compiler warnings

#ifndef INCLUDED_WARNING_DISABLE
#define INCLUDED_WARNING_DISABLE
#include <boost/config/warning_disable.hpp>
#if defined(_MSC_VER)
#pragma warning (disable: 4503) //  name length exceeded warnings
#pragma warning (disable: 4244) // "conversion from '__int64' to 'const unsigned int', possible loss of data" in Boost utree lib
#endif
#endif
