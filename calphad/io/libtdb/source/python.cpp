/*=============================================================================
 *        Copyright (c) 2012-2014 Richard Otis
 * 
 *    Distributed under the Boost Software License, Version 1.0. (See accompanying
 *    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 * =============================================================================*/

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/database.hpp"
#include "libtdb/include/logging.hpp"
#include <cmath>
#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

BOOST_PYTHON_MODULE(libtdbcpp)
{
    init_logging();
    class_<Database>("Database")
    .def(init<std::string>()) // alternative constructor
    .def("get_info", &Database::get_info)
    .def("process_command", &Database::proc_command)
    ;
}
