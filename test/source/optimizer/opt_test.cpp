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
BOOST_AUTO_TEST_CASE(SimpleIdealBinaryEquilibrium) {
	LoadDatabase("idealbin.tdb");
	double result = 0;
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
	/* It's possible for the Gibbs energy to be close but the phase compositions to be wrong.
	 * Checking that site fraction, phase fraction and mass balance constraints are satisfied
	 * along with the value of the Gibbs energy should be sufficient for most cases.
	 * This negates the need to develop an elaborate scripting system for checking every
	 * site and phase fraction on a case-by-case basis.
	 */
	BOOST_TEST_MESSAGE(PrintConditions());
	result = calculate();
	BOOST_CHECK_CLOSE_FRACTION(result, -1.00891e5, 1e-5); // from Thermo-Calc

	conditions.xfrac["NB"] = 0.6;
	conditions.statevars['T'] = 1400;
	BOOST_TEST_MESSAGE(PrintConditions());
	result = calculate();
	BOOST_CHECK_CLOSE_FRACTION(result, -9.29237e4, 1e-5); // from Thermo-Calc

	conditions.xfrac["NB"] = 0.6;
	conditions.statevars['T'] = 1000;
	BOOST_TEST_MESSAGE(PrintConditions());
	result = calculate();
	BOOST_CHECK_CLOSE_FRACTION(result, -6.33328e4, 1e-5); // from Thermo-Calc

	conditions.xfrac["NB"] = 0.01;
	conditions.statevars['T'] = 1000;
	BOOST_TEST_MESSAGE(PrintConditions());
	result = calculate();
	BOOST_CHECK_CLOSE_FRACTION(result, -5.05885e4, 1e-5); // from Thermo-Calc

	conditions.xfrac["NB"] = 0.01;
	conditions.statevars['T'] = 1200;
	BOOST_TEST_MESSAGE(PrintConditions());
	result = calculate();
	BOOST_CHECK_CLOSE_FRACTION(result, -6.48604e4, 1e-5); // from Thermo-Calc

	conditions.xfrac["NB"] = 0.99;
	conditions.statevars['T'] = 1000;
	BOOST_TEST_MESSAGE(PrintConditions());
	result = calculate();
	BOOST_CHECK_CLOSE_FRACTION(result, -4.97989e4, 1e-5); // from Thermo-Calc

	conditions.xfrac["NB"] = 0.99;
	conditions.statevars['T'] = 1200;
	BOOST_TEST_MESSAGE(PrintConditions());
	result = calculate();
	BOOST_CHECK_CLOSE_FRACTION(result, -6.39707e4, 1e-5); // from Thermo-Calc

	// LIQUID check
	conditions.xfrac["NB"] = 0.3334;
	conditions.statevars['T'] = 4000;
	BOOST_TEST_MESSAGE(PrintConditions());
	result = calculate();
	BOOST_CHECK_CLOSE_FRACTION(result, -3.63921e5, 1e-5); // from Thermo-Calc

	// SIGMA check
	conditions.xfrac["NB"] = 0.63;
	conditions.statevars['T'] = 2200;
	BOOST_TEST_MESSAGE(PrintConditions());
	result = calculate();
	BOOST_CHECK_CLOSE_FRACTION(result, -1.61081e5, 1e-5); // from Thermo-Calc
}
BOOST_AUTO_TEST_SUITE_END()
