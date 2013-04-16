// opt_helpers.cpp -- helper functions for the Gibbs energy optimizer

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/optimizer.hpp"
#include "libtdb/include/utils/math_expr.hpp"
#include <math.h>
#include <iostream>
#include <boost/current_function.hpp>


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
			// these iterator ranges should be the same size
			BOOST_THROW_EXCEPTION(internal_error()
					<< str_errinfo("Sublattice iterator ranges from database are a different size from the minimizer's internal map")
					<< specific_errinfo("std::distance(ref_subl_iter_start,ref_subl_iter_end) != std::distance(subl_iter_start,subl_iter_end)")
			);
			//std::cout << "ref size: " << std::distance(ref_subl_iter_start,ref_subl_iter_end) << std::endl;
			//std::cout << "subl size: " << std::distance(subl_iter_start,subl_iter_end) << std::endl;
		}
		for (auto i = subl_iter_start; i != subl_iter_end; ++i, ++ref_iter) {
			//std::cout << "sublattice " << std::distance(subl_iter_start,i) << std::endl;
			const auto sitefrac_begin = (*i).cbegin();
			const auto sitefrac_end = (*i).cend();
			if (std::distance(sitefrac_begin,sitefrac_end) == 0) {
				// empty sublattice?!
				BOOST_THROW_EXCEPTION(malformed_object_error() << str_errinfo("Sublattices cannot be empty"));
			}
			const double stoi_coef = (*ref_iter).stoi_coef;
			const auto sitefrac_iter = (*i).find(spec_name);
			const auto vacancy_iterator = (*i).find("VA");
			const bool pure_vacancies =
					(std::distance(sitefrac_begin,sitefrac_end) == 1 && sitefrac_begin==vacancy_iterator);
			//std::cout << sitefrac_iter->first << " stoi_coef: " << stoi_coef << std::endl;
			//std::cout << "std::distance(sitefrac_begin,sitefrac_end) == " << std::distance(sitefrac_begin,sitefrac_end) << std::endl;
			// if the sublattice is pure vacancies, don't include it in the denominator
			// unless we're calculating mole fraction of VA for some reason
			if (!pure_vacancies || spec_name == "VA") denominator += stoi_coef;
			if (sitefrac_iter != sitefrac_end) {
				const double num = stoi_coef * sitefrac_iter->second;
				numerator += num;
				//std::cout << "mole_fraction[" << spec_name << "] numerator += " << stoi_coef << " * " << num << " = " << numerator << std::endl;
				//std::cout << "mole_fraction[" << spec_name << "] denominator += " << stoi_coef * (1 - (*(*i).find("VA")).second) << " = " << denominator << std::endl;
			}
		}
		if (denominator == 0) {
			// TODO: throw an exception here
			//std::cout << "DIVIDE BY ZERO" << std::endl;
			return 0;
		}
		//std::cout << "mole_fraction = " << numerator << " / " << denominator << " = " << (numerator / denominator) << std::endl;
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
			// these iterator ranges should be the same size
			BOOST_THROW_EXCEPTION(internal_error()
					<< str_errinfo("Sublattice iterator ranges from database are a different size from the minimizer's internal map")
					<< specific_errinfo("std::distance(ref_subl_iter_start,ref_subl_iter_end) != std::distance(subl_iter_start,subl_iter_end)")
			);
			//std::cout << "ref size: " << std::distance(ref_subl_iter_start,ref_subl_iter_end) << std::endl;
			//std::cout << "subl size: " << std::distance(subl_iter_start,subl_iter_end) << std::endl;
		}
		for (auto i = subl_iter_start; i != subl_iter_end; ++i, ++ref_iter) {
			const double stoi_coef = (*ref_iter).stoi_coef;
			const auto sitefrac_begin = (*i).cbegin();
			const auto sitefrac_end = (*i).cend();
			if (std::distance(sitefrac_begin,sitefrac_end) == 0) {
				// empty sublattice?!
				BOOST_THROW_EXCEPTION(malformed_object_error() << str_errinfo("Sublattices cannot be empty"));
			}
			const auto vacancy_iterator = (*i).find("VA");
			const bool pure_vacancies =
					(std::distance(sitefrac_begin,sitefrac_end) == 1 && sitefrac_begin==vacancy_iterator);
			// if the sublattice is pure vacancies, don't include it in the denominator
			// unless we're calculating mole fraction of VA for some reason
			if (!pure_vacancies || spec_name == "VA") denominator += stoi_coef;
			if ((*i).find(spec_name) != (*i).end()) {
				////std::cout << "stoi_coef: " << stoi_coef << std::endl;
				if (std::distance(subl_iter_start,i) == deriv_subl_index && ((*i).find(deriv_spec_name) != (*i).end())) {
					numerator = stoi_coef;
				}
			}
		}
		if (denominator == 0) {
			// TODO: throw an exception here
			//std::cout << "DIVIDE BY ZERO" << std::endl;
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
		if (isvalid) { 
			double result = process_utree((*i).ast,conditions).get<double>();
			std::string subl_string;
			for (auto i = subl_config.begin(); i != subl_config.end(); ++i) {
				for (auto j = (*i).begin(); j != (*i).end(); ++j) {
					subl_string += *j + ",";
				}
				subl_string += ";";
			}
			subl_string = "[" + subl_string + "]";
			//std::cout << "get_parameter[" << phase_iter->first << "]" << subl_string << " = " << result << std::endl;
			return result;
		}
	}
	// We couldn't find a parameter
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
	//std::cout << "get_Gibbs: formation result += " << result << std::endl;

	// add energy contribution due to ideal mixing
	// + RT*y(i,s)*ln(y(i,s))
	double mixing = 0;
	// This little hack to get the number of sites in this sublattice
	// is indicative of the fact that the entire data structure for GibbsOpt
	// needs to be rewritten to conform to the way Equilibrium does it.
	auto subl_database_iter = phase_iter->second.get_sublattice_iterator();
	double total_sites = 0;
	for (auto i = subl_start; i != subl_end; ++i, ++subl_database_iter) {
		const double num_sites = (*subl_database_iter).stoi_coef;
		double sublmix = 0;
		for (auto j = (*i).begin(); j != (*i).end(); ++j) {
			if (j->second > 0) {
				sublmix += j->second * log(j->second);
			}
		}
		mixing += num_sites * sublmix;
		total_sites += num_sites;
	}
	if (total_sites <= 0) {
		BOOST_THROW_EXCEPTION(
				internal_error()
				<< str_errinfo(
						"Total number of sublattice sites is less than or equal to zero"
				)
				<< specific_errinfo(BOOST_CURRENT_FUNCTION)
		);
	}
	//std::cout << "get_Gibbs: mixing total = " << SI_GAS_CONSTANT << " * " << conditions.statevars.at('T') << " * " << mixing << std::endl;
	mixing = SI_GAS_CONSTANT * conditions.statevars.at('T') * mixing;
	result += mixing;

	// TODO: add energy contribution due to excess Gibbs energy
	result += 0;

	// Normalize result by the total number of sublattice sites
	result = (result / total_sites);
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
		BOOST_THROW_EXCEPTION(
				internal_error()
				<< str_errinfo(
						"Couldn't find sublattice in current phase"
				)
				<< specific_errinfo(BOOST_CURRENT_FUNCTION)
		);
	}
	double result = 0;
	double total_sites = 0;
	// add energy contribution due to Gibbs energy of formation (pure compounds)
	result += multiply_site_fractions_deriv(subl_start, subl_end, phase_iter, conditions, sublindex, specname);
	//std::cout << "get_Gibbs_deriv: formation result +=" << result << std::endl;

	// add energy contribution due to ideal mixing
	auto subl_find = subl_start;
	auto subl_database_iter = phase_iter->second.get_sublattice_iterator();
	const auto subl_database_iter_end = phase_iter->second.get_sublattice_iterator_end();
	while (subl_find != subl_end) {
		if (std::distance(subl_start,subl_find) == sublindex) break;
		total_sites += (*subl_database_iter).stoi_coef;
		++subl_find;
		++subl_database_iter;
	}
	// We may have broken out of the loop before we finished summing up all the sites
	// This loop finishes the summation
	for (auto i = subl_database_iter; i != subl_database_iter_end; ++i) {
		total_sites += (*i).stoi_coef;
	}
	if (subl_find == subl_end) {
		// we didn't find our sublattice
		BOOST_THROW_EXCEPTION(
				internal_error()
				<< str_errinfo(
						"Couldn't find sublattice in current phase"
				)
				<< specific_errinfo(BOOST_CURRENT_FUNCTION)
		);
	}
	if (subl_find->at(specname) > 0) {
		// number of sites for this sublattice
		// + RT * num_sites/total_sites * (1 + ln(y(specindex,sublindex)))
		const double num_sites = (*subl_database_iter).stoi_coef;
		result += SI_GAS_CONSTANT * conditions.statevars.at('T') * num_sites * (1 + log(subl_find->at(specname)));
	}

	// TODO: add excess Gibbs energy term (dGex/dy is nonzero for R-K polynomials)
	result += 0;

	// Normalize result by the total number of sublattice sites
	return (result / total_sites);
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
		////std::cout << "multiply_site_fractions: parameter = " << result << std::endl;
		/*//std::cout << "PARAMETER FOR " << phase_iter->first << " ";
		for (auto i = species.begin(); i != species.end(); ++i) {
			for (auto j = (*i).begin(); j != (*i).end(); ++j) {
				//std::cout << (*j) << ",";
			}
			//std::cout << ":";
		}
		//std::cout << " " << result << std::endl;*/
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
		//std::cout << "multiply_site_fractions: result += " << i->second << " * " << temp << std::endl;
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
	//std::cout << "multiply_site_fractions_deriv[" << specname << "]: sublindex = " << sublindex << std::endl;
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
		// get_parameter needs this information to retrieve the correct parameter
		std::vector<std::vector<std::string>> tempspecies = species;
		// We use the specieswrapper construction because interaction parameters
		//    can have multiple species in one sublattice
		//    get_parameter expects a nested vector
		std::vector<std::string> specieswrapper (1);
		specieswrapper.at(0) = i->first;
		tempspecies.push_back(specieswrapper);

		// If we are currently on the differential sublattice
		//std::cout << "species.size(): " << species.size() << std::endl;
		bool test = false;
		if (species.size() == sublindex) {
			// If we are currently on the differential species in the differential sublattice
			//std::cout << "multiply_site_fractions_deriv: i->first = " << i->first << " ; specname = " << specname << std::endl;
			if (i->first == specname) {
				return multiply_site_fractions_deriv
					(subl_next, subl_end, phase_iter, conditions, sublindex, specname, tempspecies);
			}
			else {test=true;continue;} // wrong species on the differential sublattice, it is differentiated away
		}
		assert(test==false);
		// Multiply over all combinations of the remaining site fractions
		result += i->second * multiply_site_fractions_deriv
			(subl_next, subl_end, phase_iter, conditions, sublindex, specname, tempspecies);
		//std::cout << "multiply_site_fractions_deriv result += " << i->second << " * " << (result / i->second) << std::endl;
	}
	return result;
}
