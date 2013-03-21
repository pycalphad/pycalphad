/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// structure.hpp -- database structure definitions

#ifndef INCLUDED_STRUCTURE
#define INCLUDED_STRUCTURE
#include <string>
#include <vector>
#include "libtdb/include/utils/chemical_formula.hpp"
#include "libtdb/include/utils/periodic_table.hpp"
#include "libtdb/include/conditions.hpp"
#include "libtdb/include/exceptions.hpp"
#include "libtdb/include/warning_disable.hpp"
#include <boost/spirit/home/support/utree.hpp>
#include <boost/fusion/include/adapt_struct.hpp>

struct Element {
	element_data ele_info; // data about the element (from periodic table)
	std::string ref_state; // name of stable phase at 298.15 K and 1 bar
	double mass; // mass of the pure element (g/mol)
	double H298; // enthalpy difference between 0 K and 298.15 K (SI units)
	double S298; // entropy difference between 0 K and 298.15 K (SI units)
	std::string get_name() { return ele_info.name(); }
	int atno() { return ele_info.atno(); }
};
typedef std::map<std::string, Element> Element_Collection;

class Species {
private:
	std::string spec_name;
	chemical_formula formula; // stores the amount of each element
public:
	bool operator== (const Species &other) const {
		return (spec_name == other.spec_name); // two species are equal if they have the same name
	}
	Species() { };
	Species(std::string, std::string); // stoichiometric compound (name, formula str)
	Species(std::string, chemical_formula); // stoichiometric compoound (name, formula object)
	Species(::Element); // pure element case
	std::string name() const { return spec_name; }
	chemical_formula get_formula() { return formula; }
};
typedef std::map<std::string, Species> Species_Collection;

struct Sublattice {
	double stoi_coef; // site stoichiometric coefficient
	std::vector<std::string> constituents; // list of constituents (must all be unique)
	Sublattice(double o) { stoi_coef = o; }
	Sublattice(std::vector<std::string> c) { constituents = c; }
	Sublattice(double o, std::vector<std::string> c) { stoi_coef = o; constituents = c; }
	std::vector<std::string>::const_iterator get_species_iterator() const { return constituents.cbegin(); }
	std::vector<std::string>::const_iterator get_species_iterator_end() const { return constituents.cend(); }
	void add_constituent(std::string constituent) { constituents.push_back(constituent); }
};
typedef std::vector<Sublattice> Sublattice_Collection;

struct Parameter {
	std::string phase; // name of the phase to which the parameter applies
	std::string suffix; // special indicator after underscore character: B2, A2, L12, LAVES, etc.
	std::string type; // parameter type: G, L, TC, BMAGN, etc.
	std::vector<std::vector<std::string>> constituent_array; // sublattice conditions that must be met for parameter to apply
	int degree;			// degree of Redlich-Kister term (if applicable)
	boost::spirit::utree ast; // abstract syntax tree associated with parameter (arithmetic expression with limits)
	//std::string data_ref; // scientific reference for the parameter
};
BOOST_FUSION_ADAPT_STRUCT
(
    Parameter,
    (std::string, type)
	(std::string, phase)
	(std::string, suffix)
	(std::vector<std::vector<std::string>>, constituent_array)
	(int, degree)
    (spirit::utree, ast)
	)
typedef std::vector<Parameter> Parameters;

class Phase {
private:
	std::string phase_name;
public:
	Sublattice_Collection subls; // sublattices
	Parameters params; // parameters from a database
	Phase() { };
	Phase(std::string, Sublattice_Collection); // (name, suffix, subls)
	std::string name() { return phase_name; }
	Sublattice_Collection sublattices() { return subls; } // makes a copy
	Sublattice_Collection::const_iterator get_sublattice_iterator() const { return subls.cbegin(); }
	Sublattice_Collection::const_iterator get_sublattice_iterator_end() const { return subls.cend(); }
	Parameters::const_iterator get_parameter_iterator() const { return params.cbegin(); }
	Parameters::const_iterator get_parameter_iterator_end() const { return params.cend(); }
	int sublattice_count() { return (int) subls.size(); } // number of sublattices
};
typedef std::map<std::string, Phase>   Phase_Collection;

#endif
