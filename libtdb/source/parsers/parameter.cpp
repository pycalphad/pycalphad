/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// parameter.cpp -- parser for PARAMETER command

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/database_tdb.hpp"
#include "libtdb/include/grammars/param_grammar.hpp"

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace spirit = boost::spirit;

void Database::DatabaseTDB::Parameter(std::string &argstr) {
    using boost::spirit::ascii::space;

    typedef std::string::const_iterator iterator_type;
	::Parameter ret_param;

	param_grammar parameter_grammar (macros, statevars, myspecies); 

	iterator_type iter = argstr.begin();
	iterator_type end = argstr.end();
	bool r = phrase_parse(iter, end, parameter_grammar, space, ret_param);

	if (r && iter == end)
	{
		// First, sort the species vector of each sublattice for easy later comparison.
		for (auto i = ret_param.constituent_array.begin(); i != ret_param.constituent_array.end(); ++i) {
			std::sort((*i).begin(),(*i).end());
			std::unique((*i).begin(),(*i).end()); // remove duplicate species
		}

		// We need to determine what phase this parameter corresponds to.
		// Ordering designations (L12, B2, etc.) can be in any order: e.g. L12_FCC or BCC_A2
		// When names conflict, look at the sublattice count.
		auto keyword_iter = reserved_phase_keywords.find(ret_param.phase);
		if (keyword_iter != reserved_phase_keywords.end()) {
			// the prefix is a reserved word
			// the second part must be the phase name
			// swap the strings
			ret_param.phase = ret_param.suffix;
			ret_param.suffix = keyword_iter->second;
		}
		// The phase designation is in the proper order of name_suffix now.
		// It's possible we could have two phases with the same Phase name like L12_FCC and FCC_A1.
		// We can distinguish these phases by the sublattice count.

		auto phase_iter = phases.find(ret_param.phase); // find the matching phases for the parameter

		// if that didn't work, try searching the fully qualified name
		if (ret_param.suffix.size() > 0) {
			if (phase_iter == phases.end()) 
				phase_iter = phases.find(ret_param.phase + "_" + ret_param.suffix);
			// try flipping the name around
			if (phase_iter == phases.end()) phase_iter = phases.find(ret_param.suffix + "_" + ret_param.phase);
		}
		else {
			if (phase_iter == phases.end()) {
				// No suffix listed; we can search the list of reserved keywords for a matching phase
				for (auto i = reserved_phase_keywords.begin(); i != reserved_phase_keywords.end(); ++i) {
					phase_iter = phases.find(ret_param.phase + "_" + i->second); // append a reserved suffix
					if ((phase_iter != phases.end()) 
						&& (ret_param.constituent_array.size() == (phase_iter->second).sublattice_count())) {
							// if we have a match and the sublattice counts match, then we have our phase
							break;
					}
				}
			}
		}

		// if that didn't work, maybe it's a Species parameter (chemical formula)
		// try generating a new stoichiometric phase on the fly
		// TODO: I don't think this is something that should actually be supported.
		// The example I saw might just be a mistake in the database. --rao
		/*if (phase_iter == phases.end()) {
			try {
				// try creating a new Species
				chemical_formula myform = make_chemical_formula(ret_param.phase);
				this->check_formula_validity(myform);
				myspecies[ret_param.phase] = ::Species(ret_param.phase, myform);

				// build the corresponding phase
				Sublattice_Collection subls;
				for (auto i = myform.begin(); i != myform.end(); ++i) {
					Sublattice sub(i->second); // initialize sublattice stoi_coef
					sub.constituents.push_back(myspecies[ret_param.phase]); // add species
					subls.push_back(sub);
				}
				phases[ret_param.phase] = ::Phase(ret_param.phase, subls);
				phase_iter = phases.find(ret_param.phase); // locate the new phase
			}
			catch (parse_error e) {
				// Do nothing, a parse error here isn't exceptional
			}
		}*/
		

		if (phase_iter != phases.end()) {
			if (ret_param.constituent_array.size() != (phase_iter->second).sublattice_count()) {
				std::string errmsg("Number of sublattices do not match any phase definition");
				BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(errmsg));
			}
			(phase_iter->second).params.push_back(ret_param); // add the parameter to the phase
		}
		else {
			BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo("Unknown phase: " + ret_param.phase));
		}
		//std::cout << "ADDED PARAMETER TO PHASE " << ret_param.phase << std::endl;
	}
	else
	{
		std::string::const_iterator some = iter+30;
		std::string context(iter, (some>end)?end:some);
		std::string errmsg("Syntax error: " + context + "...");
		BOOST_THROW_EXCEPTION(syntax_error() << specific_errinfo(errmsg));
		return;
	}
}
