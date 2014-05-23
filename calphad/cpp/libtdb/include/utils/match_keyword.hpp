/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// header for Thermo-Calc style abbreviated keyword matching

#ifndef INCLUDED_MATCH_KEYWORD
#define INCLUDED_MATCH_KEYWORD

#include <string>
#include <set>

std::string match_keyword(const std::string &test_string, const std::set<std::string> &keywords);

#endif
