// opt_helpers.cpp -- helper functions for the Gibbs energy optimizer

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/optimizer.hpp"
#include "libtdb/include/utils/math_expr.hpp"
#include <boost/spirit/include/support_utree.hpp>
#include <math.h>
#include <iostream>
#include <sstream>

using boost::spirit::utree;
typedef boost::spirit::utree_type utree_type;

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

// Construct the AST of the mole fraction of a species in a phase
boost::spirit::utree mole_fraction(
	const std::string &phase_name,
	const std::string &spec_name,
	const Sublattice_Collection::const_iterator ref_subl_iter_start,
	const Sublattice_Collection::const_iterator ref_subl_iter_end
	) {
		double denominator = 0;
		utree num_tree, ret_tree;

		// Iterate through all the sublattices in the phase
		for (auto i = ref_subl_iter_start; i != ref_subl_iter_end; ++i) {
			//std::cout << "sublattice " << std::distance(subl_iter_start,i) << std::endl;
			const auto sitefrac_begin = i->get_species_iterator();
			const auto sitefrac_end = i->get_species_iterator_end();
			const int sublindex = std::distance(ref_subl_iter_start,i);
			if (std::distance(sitefrac_begin,sitefrac_end) == 0) {
				// empty sublattice?!
				BOOST_THROW_EXCEPTION(malformed_object_error() << str_errinfo("Sublattices cannot be empty"));
			}
			const double stoi_coef = i->stoi_coef;
			const auto sitefrac_iter = std::find(sitefrac_begin, sitefrac_end, spec_name);
			const auto vacancy_iterator = std::find(sitefrac_begin, sitefrac_end, "VA");
			const bool pure_vacancies =
					(std::distance(sitefrac_begin,sitefrac_end) == 1 && sitefrac_begin==vacancy_iterator);

			// if the sublattice is pure vacancies, don't include it in the denominator
			// unless we're calculating mole fraction of VA for some reason
			if (!pure_vacancies || spec_name == "VA") denominator += stoi_coef;
			if (sitefrac_iter != sitefrac_end) {
				utree temp_tree;
				std::stringstream ss;
				ss << phase_name << "_" << sublindex << "_" << spec_name;
				temp_tree.push_back("*");
				temp_tree.push_back(stoi_coef);
				temp_tree.push_back(ss.str());

				if (num_tree.which() == utree_type::nil_type) num_tree.swap(temp_tree);
				else {
					utree build_tree;
					build_tree.push_back("+");
					build_tree.push_back(temp_tree);
					build_tree.push_back(num_tree);
					num_tree.swap(build_tree);
				}
			}
		}
		ret_tree.push_back("/");
		ret_tree.push_back(num_tree);
		ret_tree.push_back(denominator);

		return ret_tree;
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
