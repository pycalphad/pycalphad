/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#include "libtdb/include/database.hpp"
#include "libtdb/include/conditions.hpp"
#include "libgibbs/include/equilibrium.hpp"
#include <iostream>

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::cout << "Usage: tdbread path\n";
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
	  std::cout << maindb.get_info() << std::endl; // read out database infostring

	  // try to calculate the minimum Gibbs energy by constructing an equilibrium
	  EquilibriumFactory eqfact = EquilibriumFactory();
	  Equilibrium myeq(maindb, mainconditions, eqfact.GetIpopt());
	  // print the resulting equilibrium
	  std::cout << std::endl << myeq << std::endl;
  }
  catch (equilibrium_error &e) {
		std::string specific_info, err_msg; // error message strings
		if (std::string const * mi = boost::get_error_info<specific_errinfo>(e) ) {
			specific_info = *mi;
		}
		if (std::string const * mi = boost::get_error_info<str_errinfo>(e) ) {
			err_msg = *mi;
		}
		std::cerr << "Failed to construct equilibrium" << std::endl;
		std::cerr << "Exception: " << err_msg << std::endl;
		std::cerr << "Reason: " << specific_info << std::endl;
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
	  std::cerr << "Exception: " << err_msg << std::endl;
	  std::cerr << "Reason: " << specific_info << std::endl;
	  std::cerr << std::endl << std::endl << boost::diagnostic_information(e);
  }
  catch (std::exception &e) {
	  // last ditch effort to prevent the crash
	  std::cerr << "Exception: " << e.what() << std::endl;
  }
  return 0;

  // to test, enumerate all phases in database
  /*
  for (auto i = maindb.get_phase_iterator(); i != maindb.get_phase_iterator_end(); ++i) {
	  Phase curphase = (*i).second;
	  std::cout << curphase.name() << std::endl;
	  Sublattice_Collection subls = curphase.sublattices();
	  for (auto j = subls.begin(); j != subls.end(); ++j) {
		  std::cout << "\tSublattice " << std::distance(subls.begin(),j)+1 << " coefficient: " << (*j).stoi_coef << std::endl;
		  std::cout << "\tSublattice species: ";
		  for (auto k = (*j).constituents.begin(); k != (*j).constituents.end(); ++k) {
			  const std::string spec_name = (*k);
			  std::cout << (*k);
			  if (std::distance(k,(*j).constituents.end()) > 1) { // not yet at the last element
				  std::cout << ", ";
			  }
		  }
		  std::cout << std::endl;
	  }
  }*/
/*
  // to test, enumerate all species in database
  Species_Collection testspec = maindb.get_all_species();
  for (Species_Collection::const_iterator i = testspec.begin(); i != testspec.end(); ++i) {
	  Species curspec = (*i).second;
	  std::cout << "Species name: " << curspec.name() << std::endl;
	  chemical_formula form = curspec.get_formula();
	  for (chemical_formula::const_iterator j = form.begin(); j != form.end(); ++j) {
		  Element cur_ele = maindb.get_element((*j).first);
		  std::cout << "\t" << cur_ele.get_name() << ": " << (*j).second << std::endl;
	  }
  }*/
  // do nothing else for now
  return 0;
}
