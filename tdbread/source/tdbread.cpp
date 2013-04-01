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
  mainconditions.statevars['T'] = 1300;
  mainconditions.statevars['P'] = 101325;
  mainconditions.statevars['N'] = 1;
  mainconditions.xfrac["NB"] = 0.7;
  mainconditions.xfrac["RE"] = 0.3;
  //mainconditions.xfrac["CR"] = 0.05;
  //mainconditions.xfrac["CO"] = 0.05;
  mainconditions.elements.push_back("NB");
  mainconditions.elements.push_back("RE");
  //mainconditions.elements.push_back("CR");
  //mainconditions.elements.push_back("CO");
  mainconditions.elements.push_back("VA");
  mainconditions.phases["HCP_A3"] = true;
  mainconditions.phases["BCC_A2"] = true;
  mainconditions.phases["CHI"] = false;
  mainconditions.phases["SIGMA1"] = false;
  mainconditions.phases["LIQUID"] = true;

  // init the database by reading from the .TDB specified on the command line
  Database maindb(argv[1]);
  std::cout << maindb.get_info() << std::endl; // read out database infostring
  // calculate the minimum Gibbs energy by constructing an equilibrium
  Equilibrium myeq(maindb,mainconditions);

  // print the resulting equilibrium
  std::cout << std::endl << myeq << std::endl;
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
