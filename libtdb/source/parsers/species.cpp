// species.cpp -- parser for SPECIES command

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/database_tdb.hpp"
#include <boost/lexical_cast.hpp>

void Database::DatabaseTDB::Species(std::string &argstr) {
	std::vector<std::string> splitargs;
	boost::split(splitargs, argstr, boost::is_any_of(" "));

	if (splitargs.size() != 2) { // we have the wrong number of arguments
		std::string argnum (boost::lexical_cast<std::string>(splitargs.size())); // convert number to string
		std::string err_msg("Wrong number of arguments (" + argnum + ")");
		BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo(err_msg));
		return;
	}
	name = splitargs[0]; // set species name
	chemical_formula the_formula;
	the_formula = ::make_chemical_formula(splitargs[1]); // build the formula object

	try {
		check_formula_validity(the_formula);
	}
	catch (parse_error &e) {
		std::string offending_element;
		if (std::string const * mi = boost::get_error_info<specific_errinfo>(e) ) {
			offending_element = *mi;
		}
		std::string err_msg("Undefined element \""+ offending_element +"\" in species \"" + name + "\"");
		e << specific_errinfo(err_msg); // overwrite new specific error message (general is parse error)
		throw; // this is a showstopper, push the exception up the call stack
	}

	myspecies[name] = ::Species(name,the_formula); // add the formula to the species map
}

// Species constructor when we already have a chemical_formula
Species::Species(std::string s, std::string spec_formstr) {
	spec_name = s;
	formula = ::make_chemical_formula(spec_formstr);
}

bool Database::DatabaseTDB::check_formula_validity (chemical_formula check_formula) {
	for (chemical_formula::const_iterator j = check_formula.begin(); j != check_formula.end(); ++j) {
		::Element cur_ele = get_element((*j).first);
		if (cur_ele.get_name().empty()) { 
			BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo((*j).first)); // add element to exception info
			return false;
		}
	}
	return true; // if we made it here then everything is defined
}


// Species constructor when we already have a chemical_formula
Species::Species(std::string s, chemical_formula spec_formula) {
	spec_name = s;
	formula = spec_formula;
}
// Species constructor for the pure element case
Species::Species(::Element ele) {
	spec_name = ele.ele_info.name(); // name the species after the pure element
	formula[ele.ele_info.symbol()] = 1; // single pure element
}