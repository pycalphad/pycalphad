/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#include "libtdb/include/libtdb_pch.hpp"
#define BOOST_PYTHON_STATIC_LIB
#ifndef BOOST_SPIRIT_USE_PHOENIX_V3
#define BOOST_SPIRIT_USE_PHOENIX_V3 1
#endif
#include "libtdb/include/database.hpp"
#include "libtdb/include/conditions.hpp"
#include "libgibbs/include/equilibrium.hpp"
#include <cmath>
#include <boost/python.hpp>

using namespace boost::python;

char const* greet()
{
   return "hello, world";
}

BOOST_PYTHON_MODULE(libpytdb)
{
    class_<Database>("Database", init<std::string>())
        .def("get_info", &Database::get_info)
    ;
    def("greet", greet);
}
