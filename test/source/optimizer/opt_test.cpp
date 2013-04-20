/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// test suite for system energy minimization

#include "test/include/test_pch.hpp"
#include "test/include/fixtures/fixture_eq.hpp"
#include "libgibbs/include/equilibrium.hpp"
#include "libtdb/include/database.hpp"
#include "libtdb/include/conditions.hpp"
#define BOOST_TEST_STATIC_LINK
#include <boost/test/unit_test.hpp>

BOOST_FIXTURE_TEST_SUITE(EquilibriumSuite, EquilibriumFixture)
Database curdb = Database("idealbin.tdb");
BOOST_AUTO_TEST_CASE(SimpleIdealBinaryEquilibrium) {
	conditions.statevars['T'] = 1500;
	conditions.statevars['P'] = 101325;
	conditions.statevars['N'] = 1;
	conditions.xfrac["NB"] = 0.6;
	conditions.elements.push_back("NB");
	conditions.elements.push_back("RE");
	conditions.elements.push_back("VA");
	conditions.phases["HCP_A3"] = PhaseStatus::ENTERED;
	conditions.phases["BCC_A2"] = PhaseStatus::ENTERED;
	conditions.phases["CHI"] = PhaseStatus::ENTERED;
	conditions.phases["FCC_A1"] = PhaseStatus::ENTERED;
	conditions.phases["SIGMA1"] = PhaseStatus::ENTERED;
	conditions.phases["LIQUID"] = PhaseStatus::ENTERED;
	BOOST_CHECK_CLOSE_FRACTION(calculate(curdb),0,1e-8);
}
BOOST_AUTO_TEST_SUITE_END()
