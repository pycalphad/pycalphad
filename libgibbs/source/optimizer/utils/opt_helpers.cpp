// opt_helpers.cpp -- helper functions for the Gibbs energy optimizer

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/utils/math_expr.hpp"
#include <boost/spirit/include/support_utree.hpp>
#include <math.h>
#include <iostream>
#include <sstream>

using boost::spirit::utree;
typedef boost::spirit::utree_type utree_type;

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

				if (num_tree.which() == utree_type::invalid_type) {
					num_tree.swap(temp_tree);
				}
				else {
					utree build_tree;
					build_tree.push_back("+");
					build_tree.push_back(temp_tree);
					build_tree.push_back(num_tree);
					num_tree.swap(build_tree);
				}
			}
		}
		if (num_tree.which() != utree_type::invalid_type) {
			ret_tree.push_back("/");
			ret_tree.push_back(num_tree);
			ret_tree.push_back(denominator);
		}
		else ret_tree = utree(0);

		return ret_tree;
}
