/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Equilibrium calculations test fixture

#include "libgibbs/include/equilibrium.hpp"
#include "libtdb/include/database.hpp"
#include "libtdb/include/conditions.hpp"
#include <sstream>

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
	std::string PrintConditions() {
		std::stringstream msg;
		msg << std::endl << "Conditions: " << std::endl << "\t";
		auto sv_begin = conditions.statevars.cbegin();
		auto sv_end = conditions.statevars.cend();
		auto xf_begin = conditions.xfrac.cbegin();
		auto xf_end = conditions.xfrac.cend();
		auto el_begin = conditions.elements.cbegin();
		auto el_end = conditions.elements.cend();
		auto ph_begin = conditions.phases.cbegin();
		auto ph_end = conditions.phases.cend();
		for (auto i = sv_begin; i != sv_end; ++i) {
			msg << (*i).first << "=" << (*i).second << " ";
		}
		for (auto i = xf_begin; i != xf_end; ++i) {
			msg << "x(" << (*i).first << ")=" << (*i).second << " ";
		}
		if (el_begin != el_end) msg << std::endl << '\t';
		for (auto i = el_begin; i != el_end; ++i) {
			msg << *i << " ";
		}
		if (ph_begin != ph_end) msg << std::endl << "\t";
		for (auto i = ph_begin; i != ph_end; ++i) {
			msg << (*i).first << "=" << static_cast<char>((*i).second);
			msg << " ";
		}
		msg << std::endl;
		return msg.str();
	}
	EquilibriumFactory eqfact;
	evalconditions conditions;
	Database curdb;
	std::string dbname;
};

#endif
