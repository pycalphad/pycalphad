/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Equilibrium calculations test fixture

#include "libgibbs/include/equilibrium.hpp"
#include "libtdb/include/database.hpp"
#include "libtdb/include/conditions.hpp"

#ifndef INCLUDED_FIXTURE_EQ
#define INCLUDED_FIXTURE_EQ

struct EquilibriumFixture {
	EquilibriumFixture() : eqfact() {
	}
	void clear_conditions() {
		conditions = evalconditions();
	}
	double calculate() {
		// TODO: go beyond naive checking of the objective function
		Equilibrium testeq = Equilibrium(curdb, conditions, eqfact.GetIpopt());
		return testeq.GibbsEnergy();
	}
	void LoadDatabase(const std::string &filename) {
		if (filename != dbname) {
			// if this is a different database, load it
			curdb = Database(filename);
			dbname = filename;
		}
	}
	EquilibriumFactory eqfact;
	evalconditions conditions;
	Database curdb;
	std::string dbname;
};

#endif
