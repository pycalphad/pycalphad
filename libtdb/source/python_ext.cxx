/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#define BOOST_PYTHON_STATIC_LIB
#include <cmath>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

char const* greet()
{
   return "hello, world";
}

BOOST_PYTHON_MODULE(libpytdb)
{
    using namespace boost::python;
    def("greet", greet);
}
