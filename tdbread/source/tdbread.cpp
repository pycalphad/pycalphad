/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#ifndef BOOST_SPIRIT_USE_PHOENIX_V3
#define BOOST_SPIRIT_USE_PHOENIX_V3 1
#endif
#include "libtdb/include/database.hpp"
#include "libtdb/include/conditions.hpp"
#include "libgibbs/include/equilibrium.hpp"
#include <iostream>
#include "libtdb/include/logging.hpp"

using namespace journal;

int main(int argc, char* argv[])
{
	init_logging();
	src::severity_channel_logger< > slg(keywords::channel = "console");
	if (argc < 2)
	{
		BOOST_LOG_SEV(slg, normal) << "Usage: tdbread path\n";
		return 1;
	}
	evalconditions mainconditions;

	/*mainconditions.statevars['T'] = 1000;
  mainconditions.statevars['P'] = 101325;
  mainconditions.statevars['N'] = 1;
  mainconditions.xfrac["AL"] = 0.33;
  mainconditions.xfrac["CR"] = 0.08;
  mainconditions.elements.push_back("NI");
  mainconditions.elements.push_back("AL");
  mainconditions.elements.push_back("CR");
  mainconditions.elements.push_back("VA");
  mainconditions.phases["HCP_A3"] = PhaseStatus::ENTERED;
  mainconditions.phases["BCC_A2"] = PhaseStatus::ENTERED;
  mainconditions.phases["CHI"] = PhaseStatus::ENTERED;
  mainconditions.phases["FCC_A1"] = PhaseStatus::ENTERED;
  mainconditions.phases["SIGMA1"] = PhaseStatus::ENTERED;
  mainconditions.phases["LIQUID"] = PhaseStatus::ENTERED;
	 */

	mainconditions.statevars['T'] = 1700;
	mainconditions.statevars['P'] = 101325;
	mainconditions.statevars['N'] = 1;
	mainconditions.xfrac["NB"] = 0.02;
	mainconditions.elements.push_back("NB");
	mainconditions.elements.push_back("RE");
	mainconditions.elements.push_back("VA");
	mainconditions.phases["HCP_A3"] = PhaseStatus::ENTERED;
	mainconditions.phases["BCC_A2"] = PhaseStatus::ENTERED;
	mainconditions.phases["CHI"] = PhaseStatus::ENTERED;
	mainconditions.phases["FCC_A1"] = PhaseStatus::ENTERED;
	mainconditions.phases["SIGMA1"] = PhaseStatus::ENTERED;
	mainconditions.phases["LIQUID"] = PhaseStatus::ENTERED;

	try {
		// init the database by reading from the .TDB specified on the command line
		Database maindb(argv[1]);
		BOOST_LOG_SEV(slg, normal) << maindb.get_info(); // read out database infostring

		// try to calculate the minimum Gibbs energy by constructing an equilibrium
		EquilibriumFactory eqfact = EquilibriumFactory();
		Equilibrium myeq(maindb, mainconditions, eqfact.GetIpopt());
		// print the resulting equilibrium
		BOOST_LOG_SEV(slg, normal) << myeq;
	}
	catch (equilibrium_error &e) {
		std::string specific_info, err_msg; // error message strings
		if (std::string const * mi = boost::get_error_info<specific_errinfo>(e) ) {
			specific_info = *mi;
		}
		if (std::string const * mi = boost::get_error_info<str_errinfo>(e) ) {
			err_msg = *mi;
		}
		BOOST_LOG_SEV(slg, critical) << "Failed to construct equilibrium" << std::endl;
		BOOST_LOG_SEV(slg, critical) << "Exception: " << err_msg << std::endl;
		BOOST_LOG_SEV(slg, critical) << "Reason: " << specific_info << std::endl;
	}
	catch (boost::exception &e) {
		// catch any other uncaught Boost-enabled exceptions here
		std::string specific_info, err_msg; // error message strings
		if (std::string const * mi = boost::get_error_info<specific_errinfo>(e) ) {
			specific_info = *mi;
		}
		if (std::string const * mi = boost::get_error_info<str_errinfo>(e) ) {
			err_msg = *mi;
		}
		BOOST_LOG_SEV(slg, critical) << "Exception: " << err_msg;
		BOOST_LOG_SEV(slg, critical) << "Reason: " << specific_info;
		BOOST_LOG_SEV(slg, critical) << boost::diagnostic_information(e);
	}
	catch (std::exception &e) {
		// last ditch effort to prevent the crash
		BOOST_LOG_SEV(slg, critical) << "Exception: " << e.what();
	}
	return 0;
}
