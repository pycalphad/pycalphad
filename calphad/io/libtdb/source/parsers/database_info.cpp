/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// database_info.cpp -- parser for DATABASE_INFO command

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/database_tdb.hpp"

void Database::DatabaseTDB::Database_Info(std::string &infostr) {
	boost::replace_all(infostr,"'","\n"); // the single-quotes represent new lines in the database
	set_info(infostr);
}
