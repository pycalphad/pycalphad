/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// constituent.cpp -- parser for CONSTITUENT command

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/database_tdb.hpp"
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/finder.hpp>

void Database::DatabaseTDB::Constituent(std::string &argstr) {
	std::string phase_name, suffix;
	// get phase name
	// need to know if there's a colon in the phase name to strip away
	auto iter_range = boost::find_first(argstr, ":");
	auto spaces_range = boost::find_first(argstr, " ");
	if (iter_range.begin() < spaces_range.begin()) { // there's a colon in the first argument
		phase_name = std::string(argstr.begin(), iter_range.begin());
	}
	else {
		phase_name = std::string(argstr.begin(), spaces_range.begin());
	}

	if (phases.find(phase_name) == phases.end()) { // phase is not defined
		std::string err_msg("Phase \"" + phase_name + "\" is not defined");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}

	std::string sublstr(spaces_range.end(),argstr.end()); // phase description string
	boost::trim_if(sublstr, boost::is_any_of(":")); // trim leading/trailing colons to simplify indexing
	boost::erase_all(sublstr, " "); // remove any spaces to simplify species identification

	std::vector<std::string> subl_list; // list of sublattices
	boost::split(subl_list, sublstr, boost::is_any_of(":")); // split by sublattice
	if (subl_list.size() != phases[phase_name].sublattice_count()) {
		std::string err_msg("Number of sublattices do not match phase definition for \"" + phase_name + "\"");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}

	auto mysubls = phases[phase_name].subls.begin(); // sublattice iterator for current phase
	for (auto i = subl_list.begin(); i != subl_list.end(); ++i) {
		std::vector<std::string> species_list; // list of species in each sublattice
		boost::split(species_list, *i, boost::is_any_of(","));
		if (species_list.empty()) {
			std::string err_msg("Syntax error in sublattice descripton");
			BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
		}

		for (auto j = species_list.begin(); j != species_list.end(); ++j) {
			if (boost::ends_with(*j,"%")) {
				// TODO: this is a major constituent in this sublattice
				// we need to indicate that somehow
				boost::erase_tail(*j,1); // remove the % marker from the species name
			}
			if (myspecies.find(*j) == myspecies.end()) { // Species does not exist
				std::string err_msg("Undefined species \"" + *j + "\" in sublattice description");
				BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
			}
			// check for duplicates
			for (auto k = j+1; k != species_list.end(); ++k) {
				if (*k == *j) {
					// non-unique species
					std::string err_msg("Duplicate species \"" + *j + "\" in sublattice description");
					BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
				}
			}
			(*mysubls).add_constituent(*j); // add species as sublattice constituent
		}
		std::sort((*mysubls).constituents.begin(),(*mysubls).constituents.end()); // sort the species in the sublattice
		std::unique((*mysubls).constituents.begin(),(*mysubls).constituents.end()); // remove duplicates
		++mysubls; // go to next sublattice
		// we already verified mysubls and subl_list are the same size, so this is safe
	}
}
