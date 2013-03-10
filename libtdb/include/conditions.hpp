// conditions.h -- header file for thermodynamic state variable declaration

#ifndef INCLUDED_CONDITIONS
#define INCLUDED_CONDITIONS
#include <map>
#include <vector>

struct evalconditions { 
std::map<char,double> statevars; // state variable values
std::vector<std::string> elements; // elements under consideration
std::map<std::string,double> xfrac; // system mole fractions
};

#define SI_GAS_CONSTANT 8.3144621 // J/mol-K

#endif