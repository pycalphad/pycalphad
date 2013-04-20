/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#ifndef MESH_INCLUDED
#define MESH_INCLUDED

// declaration for Mesh object

#include "libtdb/include/conditions.hpp"

class Mesh {
private:
	// collection of evalconditions objects
	const evalconditions startpoint;
public:
	Mesh(const evalconditions&);
};

#endif
