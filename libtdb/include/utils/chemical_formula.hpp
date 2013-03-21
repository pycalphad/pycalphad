/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// chemical_formula.hpp -- header file for chemical formula declaration

#ifndef CHEMICAL_FORMULA_INCLUDED
#define CHEMICAL_FORMULA_INCLUDED

#include <string>
#include <map>

typedef std::map<std::string,double> chemical_formula; // maps atomic symbol of element to stoichiometric coefficient

chemical_formula make_chemical_formula (std::string); // defined in chemical_formula.cpp

#endif
