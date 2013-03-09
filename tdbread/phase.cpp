// phase.cpp -- parser for PHASE command

#include "stdafx.h"
#include "database_tdb.h"
#include <boost/lexical_cast.hpp>

// TODO: add support for the optional auxillary info string (comes after last sublattice argument)
void Database::DatabaseTDB::Phase(std::string &argstr) {
	std::string name; // phase name
	std::string codes, phase_code; // character codes for the phase
	int num_subl; // number of sublattices
	Sublattice_Collection subls; // sublattices, we only init stoi_coef in this parser
	std::vector<std::string> splitargs;

	boost::split(splitargs, argstr, boost::is_any_of(" "));
	if (splitargs.size() < 4) { // we have the wrong number of arguments
		std::string argnum (boost::lexical_cast<std::string>(splitargs.size())); // convert number to string
		std::string err_msg("Wrong number of arguments (" + argnum + ")");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}
	auto i = splitargs.begin(); // generate an iterator
	// the first few arguments are not in a loop
	// we increment the iterator manually
	// now evaluating the name argument
	auto iter_range = boost::find_first(*i, ":");
	name = std::string((*i).begin(),iter_range.begin()); // name is everything before the (optional) colon
	phase_code = std::string(iter_range.end(),(*i).end()); // TODO: check for GES phase-type codes after the colon

	if (phase_code.length() > 1) { // can be one character or zero characters
		std::string err_msg("Phase name incorrectly formatted");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}
	if (phases.find(name) != phases.end()) { // the phase already exists
		std::string err_msg("Phase \"" + name + "\" already defined");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}

	++i;
	// now at phase character code argument

	codes = *i;
	++i;
	// now at number of sublattices argument

	try {
		num_subl = boost::lexical_cast<int>(*i);
		++i;
		// now at first site stoi_coef argument
	}
	catch (boost::bad_lexical_cast e) {
		std::string err_msg ("Non-integer input for integer parameter");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}
	if (splitargs.size() != (num_subl+3)) { // we have the wrong number of arguments
		std::string argnum (boost::lexical_cast<std::string>(splitargs.size())); // convert number to string
		std::string err_msg("Wrong number of arguments (" + argnum + ") for " + *i + " sublattice" + ((num_subl != 1) ? "s" : ""));
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}
	// collect all of the site occupancies
	try {
		while (i != splitargs.end()) {
			double stoi_coef = boost::lexical_cast<double>(*i);
			if (stoi_coef == 0) {
				std::cout << "0 stoi_coef, original was: " << *i << std::endl;
			}
			Sublattice s(stoi_coef); // init sublattice with site stoi_coef
			subls.push_back(s); // add sublattice to collection
			++i;
		}
	}
	catch (boost::bad_lexical_cast e) {
		std::string err_msg ("Non-numeric input for numeric parameter");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}
	// TODO: read the character codes for the phase
	for (auto j = codes.begin(); j != codes.end(); ++j) {
	}
	phases[name] = ::Phase(name, subls); // add Phase to the Database
}

Phase::Phase (std::string phasename, Sublattice_Collection s) {
	phase_name = phasename;
	subls = s;
}