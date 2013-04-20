/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// naive partition of phase space without any adaptive methods

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/equilibrium.hpp"
#include "libgibbs/include/mesh.hpp"
#include "libtdb/include/database.hpp"
#include "libtdb/include/exceptions.hpp"
#include "libtdb/include/conditions.hpp"

void NaivePartition(const Database &DB, const EquilibriumFactory &eqfact, const Mesh &eqmesh) {
	// set up all equilibria calculations on the EquilibriumFactory queue
	// run all calculations
	// get phase_map data and associate it with the conditions specified at each point (Boost MultiIndex?)

}
