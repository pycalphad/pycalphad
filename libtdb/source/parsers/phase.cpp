/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// phase.cpp -- parser for PHASE command

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/database_tdb.hpp"
#include "libtdb/include/utils/match_keyword.hpp"
#include <boost/lexical_cast.hpp>

// TODO: add support for the optional auxillary info string (comes after last sublattice argument)
void Database::DatabaseTDB::Phase(std::string &argstr) {
	std::string name; // phase name
	std::string codes, phase_code; // character codes for the phase
	int num_subl; // number of sublattices
	Sublattice_Collection subls; // sublattices, we only init stoi_coef in this parser
	std::vector<std::string> splitargs, init_commands;

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
	catch (boost::bad_lexical_cast &) {
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
	catch (boost::bad_lexical_cast &) {
		std::string err_msg ("Non-numeric input for numeric parameter");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}
	// Ensure that the type definitions for the phase have been defined
	for (auto j = codes.cbegin(); j != codes.cend(); ++j) {
		const std::string type(j, j+1);
		auto typefind = type_definitions.find(type);
		if (typefind == type_definitions.end()) {
			// undefined type definition
			std::string err_msg ("Undefined type definition: ");
			BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg + type));
		}
		else {
			init_commands.push_back(typefind->second); // add command to list
		}
	}
	phases[name] = ::Phase(name, subls, init_commands); // add Phase to the Database
}

void Phase::modify_phase(const std::string &command) {
	std::set<std::string> keywords = {
			"AMEND_PHASE_DESCRIPTION",
			"DISORDERED_PART",
			"MAGNETIC_ORDERING"
	};
	std::vector<std::string> words;
	std::string matching_command;
	boost::split(words, command, boost::is_any_of(" "), boost::token_compress_on);

	matching_command = match_keyword(words[0], keywords);

	if (matching_command == "AMEND_PHASE_DESCRIPTION") {
		// Modify a phase description
		if (words.size() >= 3) {
			auto words_iter = words.begin(); // on A_P_D command
			++words_iter; // name of phase
			if (*words_iter != phase_name) {
				// phase name mismatch
				std::string errmsg
				("Phase name mismatch in AMEND_PHASE_DESCRIPTION type definition for phase " + phase_name);
				BOOST_THROW_EXCEPTION(syntax_error() << specific_errinfo(errmsg));
			}
			++words_iter; // specific amendment to make
			std::string matching_subcommand = match_keyword(*words_iter, keywords);
			if (matching_subcommand == "MAGNETIC_ORDERING") {
				if (words.size() >= 5) {
					++words_iter; // AFM factor
					double afm_factor = boost::lexical_cast<double>(*words_iter);
					++words_iter; // SRO enthalpy factor
					double sro_factor = boost::lexical_cast<double>(*words_iter);

					// Modify the phase's magnetic settings
					magnetic_afm_factor = afm_factor;
					magnetic_sro_enthalpy_order_fraction = sro_factor;
					return;
				}
			}
			else if (matching_subcommand == "DISORDERED_PART") {
				// TODO: Atomic ordering model not yet implemented
				return;
			}
			BOOST_THROW_EXCEPTION(syntax_error() << specific_errinfo("Not implemented: " + command));
		}

	}
	BOOST_THROW_EXCEPTION(syntax_error() << specific_errinfo("Improper syntax for AMEND_PHASE_DESCRIPTION type definition in " + phase_name));
}

void Phase::process_type_definition(const std::string &command) {
	std::set<std::string> keywords = {
			"GES",
			"SEQ"
	};
	std::string test_command = std::string(command.begin(), boost::find_first(command, " ").begin());
	std::string matching_command = match_keyword(test_command, keywords);

	if (matching_command == "GES") {
		std::string remaining_command(boost::find_first(command, " ").end(), command.end());
		modify_phase(remaining_command);
	}
	else if (matching_command == "SEQ") {
		// we don't implement any of the optimizations provided by this command
	}
	else BOOST_THROW_EXCEPTION(syntax_error() << specific_errinfo("Unknown command in type definition for phase " + phase_name));
}

Phase::Phase (std::string phasename, Sublattice_Collection s, std::vector<std::string> init_commands) {
	phase_name = phasename;
	subls = s;
	magnetic_afm_factor = 0;
	magnetic_sro_enthalpy_order_fraction = 0;
	init_cmds = init_commands;
	// Apply all of this phase's type definitions
	for (auto i = init_cmds.begin(); i != init_cmds.end(); ++i) process_type_definition(*i);
}
