/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Thermo-Calc-style keyword matching for abbreviations

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/exceptions.hpp"
#include <boost/algorithm/string.hpp>
#include <set>

// Is test an abbreviated form of fullword?
bool is_abbreviation_of (const std::string &fullword, const std::string &test) {
	if (boost::istarts_with(fullword, test)) return true; // trivial case
	std::vector<std::string> test_splitargs, full_splitargs;
	boost::split(full_splitargs, fullword, boost::is_any_of("_-")); // split by the separators
	boost::split(test_splitargs, test, boost::is_any_of("_-"));
	auto test_iter = test_splitargs.begin();
	auto full_iter = full_splitargs.begin();
	const auto test_end = test_splitargs.end();
	const auto full_end = full_splitargs.end();
	while ((test_iter != test_end) || (full_iter != full_end)) {
		if (!boost::istarts_with(*full_iter, *test_iter)) break;
		++test_iter;
		++full_iter;
	}
	if (test_iter == test_end) return true; // Test string never failed a comparison and reached the end
	else return false;
}

// Return the full name of a Thermo-Calc-style abbreviated keyword
// Will throw if ret_strings.size() > 1 ==> ambiguous command
// Will throw if ret_strings.size() == 0 ==> no match
std::string match_keyword(const std::string &test_string, const std::set<std::string> &keywords) {
	std::vector<std::string> ret_strings;
	for (auto i = keywords.begin(); i != keywords.end(); ++i) {
		if (is_abbreviation_of(*i, test_string)) ret_strings.push_back(*i);
	}
	if (ret_strings.size() > 1) {
		std::string errstring;
		errstring = "Ambiguous command " + test_string + ". Possible matches: ";
		for (auto i = ret_strings.begin(); i != ret_strings.end(); ++i) {
			errstring += *i;
			errstring += " ";
		}
		BOOST_THROW_EXCEPTION(syntax_error() << specific_errinfo(errstring));
	}
	if (ret_strings.size() == 0) {
		std::string errstring;
		errstring = "Unknown command " + test_string;
		BOOST_THROW_EXCEPTION(syntax_error() << specific_errinfo(errstring));
	}
	return ret_strings[0];
}
