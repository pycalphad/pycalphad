// opt_helpers.cpp -- helper functions for the Gibbs energy optimizer
#include "optimizer.h"
#include "math_expr.h"
#include <math.h>


double mole_fraction(
	const std::string &spec_name,
	const Sublattice_Collection::const_iterator ref_subl_iter_start,
	const Sublattice_Collection::const_iterator ref_subl_iter_end,
	const sublattice_vector::const_iterator subl_iter_start,
	const sublattice_vector::const_iterator subl_iter_end
	) {
		auto ref_iter = ref_subl_iter_start;
		double numerator = 0;
		double denominator = 0;
		if (std::distance(ref_subl_iter_start,ref_subl_iter_end) != std::distance(subl_iter_start,subl_iter_end)) {
			// these iterators should have ranges of the same size
			// TODO: throw an exception here
			std::cout << "ref size: " << std::distance(ref_subl_iter_start,ref_subl_iter_end) << std::endl;
			std::cout << "subl size: " << std::distance(subl_iter_start,subl_iter_end) << std::endl;
		}
		if (spec_name == "VA") {
			return 0; // mole fraction of vacancies are always zero
		}
		for (auto i = subl_iter_start; i != subl_iter_end; ++i) {
			double stoi_coef = (*ref_iter).stoi_coef;
			//std::cout << "stoi_coef: " << stoi_coef << std::endl;
			numerator += stoi_coef * (*(*i).find(spec_name)).second;
			denominator += stoi_coef * (1 - (*(*i).find("VA")).second);
			std::cout << "mole_fraction numerator += " << stoi_coef << " * " << (*(*i).find(spec_name)).second << std::endl;
			std::cout << "mole_fraction denominator += " << stoi_coef << " * " << (1 - (*(*i).find("VA")).second) << std::endl;
			++ref_iter;
		}
		if (denominator == 0) {
			// TODO: throw an exception here
			std::cout << "DIVIDE BY ZERO" << std::endl;
			return 0;
		}
		return (numerator / denominator);
}

double mole_fraction_deriv(
	const std::string &spec_name,
	const std::string &deriv_spec_name,
	const int &deriv_subl_index,
	const Sublattice_Collection::const_iterator ref_subl_iter_start,
	const Sublattice_Collection::const_iterator ref_subl_iter_end,
	const sublattice_vector::const_iterator subl_iter_start,
	const sublattice_vector::const_iterator subl_iter_end
	) {
		auto ref_iter = ref_subl_iter_start;
		double numerator = 0;
		double denominator = 0;
		if (std::distance(ref_subl_iter_start,ref_subl_iter_end) != std::distance(subl_iter_start,subl_iter_end)) {
			// these iterators should have ranges of the same size
			// throw an exception here
			std::cout << "ref size: " << std::distance(ref_subl_iter_start,ref_subl_iter_end) << std::endl;
			std::cout << "subl size: " << std::distance(subl_iter_start,subl_iter_end) << std::endl;
		}
		if (spec_name == "VA" || deriv_spec_name == "VA") {
			return 0; // mole fraction of vacancies are always zero
		}
		for (auto i = subl_iter_start; i != subl_iter_end; ++i) {
			double stoi_coef = (*ref_iter).stoi_coef;
			//std::cout << "stoi_coef: " << stoi_coef << std::endl;
			if (std::distance(subl_iter_start,i) == deriv_subl_index) {
			numerator = stoi_coef;
			}
			denominator += stoi_coef * (1 - (*(*i).find("VA")).second);
			++ref_iter;
		}
		if (denominator == 0) {
			// TODO: throw an exception here
			std::cout << "DIVIDE BY ZERO" << std::endl;
			return 0;
		}
		return (numerator / denominator);
}

// find the Parameter corresponding to the current conditions and calculate its value based on conditions
double get_parameter
	(
	const Phase_Collection::const_iterator phase_iter,
	const evalconditions &conditions,
	const std::vector<std::vector<std::string>> &subl_config,
	const std::string type
	) {
	// Search the phase's parameters
	for (auto i = phase_iter->second.get_parameter_iterator(); 
		i != phase_iter->second.get_parameter_iterator_end(); ++i) {
		if ((*i).type != type) continue; // skip if not the parameter type of interest
		if (subl_config.size() != (*i).constituent_array.size()) continue; // skip if sublattice counts do not match
		// We cannot do a direct comparison of the nested vectors because
		//    we must check the case where the wildcard '*' is used in some sublattices in the parameter
		// TODO: this would be a nice thing to write a function for in the Sublattice_Collection object
		bool isvalid = true;
		auto array_begin = (*i).constituent_array.begin();
		auto array_iter = array_begin;
		auto array_end = (*i).constituent_array.end();
		while (array_iter != array_end) {
			const std::string firstelement = *(*array_iter).cbegin();
			// if the parameter sublattices don't match, or the parameter isn't using a wildcard, do not match
			if (!(
				(*array_iter) == subl_config[std::distance(array_begin,array_iter)]) 
				|| (firstelement == "*")
				) {
				isvalid = false;
				break;
			}
			++array_iter;
		}
		if (isvalid) return process_utree((*i).ast,conditions).get<double>();
	}
	// TODO: We couldn't find a parameter. We should probably throw an exception here
	return 0;
}

// calculate the Gibbs energy based on the sublattice model
double get_Gibbs
	(
	const sublattice_vector::const_iterator subl_start,
	const sublattice_vector::const_iterator subl_end,
	const Phase_Collection::const_iterator phase_iter,
	const evalconditions &conditions
	) {

	double result = 0;
	
	// add energy contribution due to Gibbs energy of formation (pure compounds)
	result += multiply_site_fractions(subl_start, subl_end, phase_iter, conditions);
	std::cout << "get_Gibbs: formation result += " << result << std::endl;

	// add energy contribution due to ideal mixing
	// + RT*y(i,s)*ln(y(i,s))
	double mixing = 0;
	for (auto i = subl_start; i != subl_end; ++i) {
		for (auto j = (*i).begin(); j != (*i).end(); ++j) {
			if (j->second > 0) {
				mixing += j->second * log(j->second);
				std::cout << "get_Gibbs(" << std::distance(subl_start,i) << ")(" << std::distance((*i).begin(),j) << "): mixing += " << j->second << " * " << log(j->second) << std::endl;
			}
		}
	}
	std::cout << "get_Gibbs: mixing total = " << SI_GAS_CONSTANT << " * " << conditions.statevars.at('T') << " * " << mixing << std::endl;
	mixing = SI_GAS_CONSTANT * conditions.statevars.at('T') * mixing;
	result += mixing;

	// TODO: add energy contribution due to excess Gibbs energy
	result += 0;

	return result;
}

// calculate dG/dy(l,s,j)
double get_Gibbs_deriv
	(
	const sublattice_vector::const_iterator subl_start,
	const sublattice_vector::const_iterator subl_end,
	const Phase_Collection::const_iterator phase_iter,
	const evalconditions &conditions,
	const int &sublindex,
	const std::string &specname
	) {
		if (std::distance(subl_start,subl_end) < sublindex) {
			// out of bounds index
			// TODO: throw an exception
		}
	double result = 0;
	// add energy contribution due to Gibbs energy of formation (pure compounds)
	result += multiply_site_fractions_deriv(subl_start, subl_end, phase_iter, conditions, sublindex, specname);
	std::cout << "get_Gibbs_deriv: formation result +=" << result << std::endl;

	// add energy constribution due to ideal mixing
	auto subl_find = subl_start;
	while (subl_find != subl_end) {
		if (std::distance(subl_start,subl_find) == sublindex) break;
		++subl_find;
	}
	if (subl_find == subl_end) {
		// TODO: throw an exception
		// we didn't find our sublattice
		std::cout << "get_Gibbs_deriv: couldn't find the corresponding sublattice for mixing parameter" << std::endl;
	}
	// + RT * (1 + ln(y(specindex,sublindex)))
	if (subl_find->at(specname) > 0) { 
	std::cout << "get_Gibbs_deriv: mixing result += " << SI_GAS_CONSTANT << "*" << conditions.statevars.at('T') << "*" << "(1 + log(" << subl_find->at(specname) << "))" << std::endl;
	result += SI_GAS_CONSTANT * conditions.statevars.at('T') * (1 + log(subl_find->at(specname))); 
	}

	// TODO: add excess Gibbs energy term (dGex/dy is nonzero for R-H polynomials)
	result += 0;
	//std::cout << "dGdy(" << phase_iter->first << ")(" << sublindex << ")(" << specname << "): " << result << std::endl;
	return result;
	// + RT*y(i,s)*ln(y(i,s))
}

// recursive function to handle arbitrary permutations of site fractions
double multiply_site_fractions
	(
	const sublattice_vector::const_iterator subl_start,
	const sublattice_vector::const_iterator subl_end,
	const Phase_Collection::const_iterator phase_iter,
	const evalconditions &conditions,
	std::vector<std::vector<std::string>> species
	) {

	double result = 0;
	if (subl_start == subl_end) {
		// We are at the bottom of the recursive loop
		// No sublattices left
		// return the corresponding Gibbs parameter
		result = get_parameter(phase_iter, conditions, species);
		//std::cout << "multiply_site_fractions: parameter = " << result << std::endl;
		/*std::cout << "PARAMETER FOR " << phase_iter->first << " ";
		for (auto i = species.begin(); i != species.end(); ++i) {
			for (auto j = (*i).begin(); j != (*i).end(); ++j) {
				std::cout << (*j) << ",";
			}
			std::cout << ":";
		}
		std::cout << " " << result << std::endl;*/
		return result;
	}

	// Iterate over the first sublattice only
	// Iterate through each species
	for (auto i = (*subl_start).cbegin(); i != (*subl_start).cend(); ++i) {
		// Add the current species to the sublattice configuration: e.g. NI:AL:VA
		// get_Gibbs needs this information to retrieve the correct parameter
		std::vector<std::vector<std::string>> tempspecies = species;
		// We use the specieswrapper construction because interaction parameters
		//    can have multiple species in one sublattice
		//    get_parameter expects a nested vector
		std::vector<std::string> specieswrapper (1);
		specieswrapper.at(0) = i->first;
		tempspecies.push_back(specieswrapper);
		// Multiply over all combinations of the remaining site fractions
		double temp = multiply_site_fractions(subl_start+1, subl_end, phase_iter, conditions, tempspecies);
		std::cout << "multiply_site_fractions: result += " << i->second << " * " << temp << std::endl;
		result += i->second * temp;
	}
	return result;
}

// recursive function to handle arbitrary permutations of site fractions for dG/dy
double multiply_site_fractions_deriv
	(
	const sublattice_vector::const_iterator subl_start,
	const sublattice_vector::const_iterator subl_end,
	const Phase_Collection::const_iterator phase_iter,
	const evalconditions &conditions,
	const int &sublindex,
	const std::string &specname,
	std::vector<std::vector<std::string>> species
	) {
	sublattice_vector::const_iterator subl_next = subl_start;
	++subl_next;
	double result = 0;
	if (subl_start == subl_end) {
		// We are at the bottom of the recursive loop
		// No sublattices left
		// return the corresponding Gibbs parameter
		result = get_parameter(phase_iter, conditions, species);
		//std::cout << "multiply_site_fractions_deriv: parameter = " << result << std::endl;
		return result;
	}


	// Iterate over the first sublattice only
	// Iterate through each species
	for (auto i = (*subl_start).cbegin(); i != (*subl_start).cend(); ++i) {
		// Add the current species to the sublattice configuration: e.g. NI:AL:VA
		// get_Gibbs needs this information to retrieve the correct parameter
		std::vector<std::vector<std::string>> tempspecies = species;
		// We use the specieswrapper construction because interaction parameters
		//    can have multiple species in one sublattice
		//    get_parameter expects a nested vector
		std::vector<std::string> specieswrapper (1);
		specieswrapper.at(0) = i->first;
		tempspecies.push_back(specieswrapper);

		// If we are currently on the differential sublattice
		if ((species.size() - 1) == sublindex) {
			// If we are currently on the differential species in the differential sublattice
			if (i->first == specname) {
				return multiply_site_fractions_deriv
					(subl_next, subl_end, phase_iter, conditions, sublindex, specname, tempspecies);
			}
			else return 0; // wrong species on the differential sublattice, it is differentiated away
		}
		// Multiply over all combinations of the remaining site fractions
		result += i->second * multiply_site_fractions_deriv
			(subl_next, subl_end, phase_iter, conditions, sublindex, specname, tempspecies);
	}
	return result;
}