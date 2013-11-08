/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// parser for TYPE_DEFINITION command

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/database_tdb.hpp"
#include "libtdb/include/logging.hpp"
#include <boost/algorithm/string/find.hpp>
#include <string>

void Database::DatabaseTDB::Type_Definition(std::string &argstr) {
	logger db_log(journal::keywords::channel = "data");
	auto typefind = boost::find_first(argstr, " "); // contains the iter range containing the character of the typedef
	std::string type (argstr.begin(), typefind.begin()); // Example: % , (
	std::string command (typefind.end(), argstr.end()); // Example: GES A_P_D FCC_A1 MAGNETIC  -3.0    2.80000E-01
	if (type_definitions.find(type) != type_definitions.end()) {
		BOOST_LOG_SEV(db_log, critical) << "Illegal attempt to redefine type definition for " << type;
		BOOST_THROW_EXCEPTION(parse_error() << str_errinfo("Type already defined") << specific_errinfo(type));
	}
	type_definitions[type] = command;
}
