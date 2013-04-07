/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// chemical_formula.cpp -- utils for building chemical_formula objects
// TODO: rewrite this using Spirit so I can handle the case of NIHF, NI5HF correctly

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/utils/chemical_formula.hpp"
#include "libtdb/include/exceptions.hpp"
#include <boost/lexical_cast.hpp>

// formstr is a formula string of form, e.g., AL1O1H1CL2H6O3
// elements can be grouped separately
chemical_formula make_chemical_formula (std::string formstr) {
	bool matching_element = false; // flag for matching element symbols versus numbers
	std::string ele_sym; // currently matched element symbol
	std::string::const_iterator lastpos = formstr.begin(); // start position for current copy
	chemical_formula build_formula; // chemical formula we are building

	// invariant: processed from the beginning to position pos in formstr
	for (std::string::const_iterator pos = lastpos; pos != formstr.end(); pos++) {
		if (!matching_element) {
			if (isalpha(*pos) || *pos == '/' || *pos == '-' || *pos == '+') {
				// current character is part of an element name
				// we are not matching an element, so copy the current number into the formula
				if (!ele_sym.empty()) {
					double stoi_coef;
					try {
						stoi_coef = boost::lexical_cast<double>(std::string(lastpos,pos));
					}
					catch (boost::bad_lexical_cast &e) {
						std::string err_msg ("Invalid stoichiometric coefficient: " + std::string(lastpos,pos));
						BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
					}
					build_formula[ele_sym] = build_formula[ele_sym] + stoi_coef; // increase the coefficient

					lastpos = pos; // advance the last position
					ele_sym.erase(); // reset current element name
					matching_element = true;
					continue;
				}
				else {
					// new element to be matched
					lastpos = pos; // advance the last position
					matching_element = true;
					continue;
				}
			}
			else if (isdigit(*pos) || *pos == '.') {
				// current character is part of a stoichiometric coefficient
				// we are matching numbers, so keep on going
				continue;
			}
			else {
				// we have an invalid character
				std::string err_msg ("Invalid character: " + boost::lexical_cast<std::string>(*pos));
				BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
			}
		}
		else {
			// we must be matching a number currently
			if (isalpha(*pos) || *pos == '/' || *pos == '-' || *pos == '+') {
				// current character is part of an element name
				// we are matching an element, so keep on going
				continue;
			}
			else if (isdigit(*pos) || *pos == '.') {
				// current character is part of a stoichiometric coefficient
				// we are matching element symbols, so save the name of the element so far
				ele_sym = std::string(lastpos,pos);
				//std::cout << "New element: " << ele_sym << std::endl;
				lastpos = pos; // advance the last position
				matching_element = false;
				continue;
			}
			else {
				// we have an invalid character
				std::string err_msg ("Invalid character: " + boost::lexical_cast<std::string>(*pos));
				BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
			}
		}
	}
	// but wait... there's still one more token left to process
	if (!matching_element) {
		std::string::const_iterator pos = formstr.end(); // we know we're at the end of the string
		// we are matching a number
		// end of the string, so let's add this last coefficient
		if (!ele_sym.empty()) {
			double stoi_coef;
			try {
				stoi_coef = boost::lexical_cast<double>(std::string(lastpos,pos));
			}
			catch (boost::bad_lexical_cast &e) {
				std::string err_msg ("Invalid stoichiometric coefficient: " + std::string(lastpos,pos));
				BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
			}
			build_formula[ele_sym] = build_formula[ele_sym] + stoi_coef; // increase the coefficient
		}
	}
	else {
		// we shouldn't be here
		// formulae should always end in a coefficient
		std::string err_msg ("Invalid format for chemical formula");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
	}
	return build_formula; // return the built object
}
