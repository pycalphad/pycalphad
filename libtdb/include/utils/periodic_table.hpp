/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// periodic_table.hpp -- header file for periodic table declaration

#include <string>
#include <map>

// the element_data class contains purely chemical information about the elements
class element_data {
private:
	std::string ele_sym; // symbol of the element
	std::string fullname; // name of the element
	int atomic_number;
public:
	element_data() { };
	element_data(std::string sym, std::string name, int atno) {
		ele_sym = sym;
		fullname = name;
		atomic_number = atno;
	}
	std::string name() { return fullname; }
	std::string symbol() { return ele_sym; }
	int atno() { return atomic_number; }
};

// the periodic_table_elements map contains the periodic table
std::map<std::string,element_data> create_periodic_table(); // defined in periodic_table.cpp
extern std::map<std::string,element_data> periodic_table_elements; // defined in database_tdb.cpp
