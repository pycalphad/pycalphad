// chemical_formula.h -- header file for chemical formula declaration

#ifndef CHEMICAL_FORMULA_INCLUDED
#define CHEMICAL_FORMULA_INCLUDED

#include <string>
#include <map>

typedef std::map<std::string,double> chemical_formula; // maps atomic symbol of element to stoichiometric coefficient

chemical_formula make_chemical_formula (std::string); // defined in chemical_formula.cpp

#endif