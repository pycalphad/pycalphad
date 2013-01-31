// element.cpp -- parser for ELEMENT command
#include "database_tdb.h"
#include <boost/lexical_cast.hpp>

void Database::Element(std::string &argstr) {
	std::vector<std::string> splitargs;
	boost::split(splitargs, argstr, boost::is_any_of(" "));
	if (splitargs.size() != 5) { // we have the wrong number of arguments
		std::string argnum (boost::lexical_cast<std::string>(splitargs.size())); // convert number to string
		std::string err_msg("Wrong number of arguments (" + argnum + ")");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}
	if (splitargs[0].size() > 2) { // elements only have 2 or less characters
		std::string err_msg("Bad element declaration \"" + splitargs[0] + "\"");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}
	// TODO: temporary until the info handling gets added for these
	std::string elementname = splitargs[0];
	std::string refphase = splitargs[1]; // stable phase at 298.15K and 1 bar
	double mass; // mass of pure element (g/mol)
	double H298;
	double S298;
	try {
		mass = boost::lexical_cast<double>(splitargs[2]);
		H298 = boost::lexical_cast<double>(splitargs[3]);
		S298 = boost::lexical_cast<double>(splitargs[4]);
	}
	catch (boost::bad_lexical_cast e) {
		std::string err_msg ("Non-numeric input for numeric parameter");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}
	if (elements.find(elementname) == elements.end()) { // has the element not already been defined?
		element_data *cur_ele = &periodic_table_elements[elementname];
		if (!(*cur_ele).name().empty()) { // does the element exist on the periodic table?
			elements[elementname].ele_info = *cur_ele;
			elements[elementname].mass = mass;
			elements[elementname].H298 = H298;
			elements[elementname].S298 = S298;
			elements[elementname].ref_state = refphase;
		}
		else {
			std::string err_msg ("Element \"" + elementname + "\" does not exist in periodic table");
			BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
		}
	}
	else {
		std::string err_msg ("Element already defined");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}
	// init corresponding Species object for the pure element
	if(myspecies[elementname].name().empty()) { // does the species not yet exist?
		myspecies[elementname] = ::Species(get_element(elementname)); // create pure element species
	}
	else {
		std::string err_msg ("Species name collision");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}
}