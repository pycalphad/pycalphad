/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// calculation of the first derivative of the Gibbs energy function

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/optimizer.hpp"

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
				<< str_errinfo("Couldn't find sublattice in current phase")
				<< specific_errinfo("Sublattice index is out of bounds")
		);
	}
	double result = 0;
	double total_sites = 0;
	double total_mixing_sites = 0;
	// add energy contribution due to Gibbs energy of formation (pure compounds)
	result += multiply_site_fractions_deriv(subl_start, subl_end, phase_iter, conditions, sublindex, specname);
	//std::cout << "get_Gibbs_deriv: formation result +=" << result << std::endl;

	// add energy contribution due to ideal mixing
	auto subl_find = subl_start;
	auto subl_database_iter = phase_iter->second.get_sublattice_iterator();
	const auto subl_database_iter_end = phase_iter->second.get_sublattice_iterator_end();
	while (subl_find != subl_end) {
		if (std::distance(subl_start,subl_find) == sublindex) break;
		int speccount = 0;
		total_sites += (*subl_database_iter).stoi_coef;
		const auto spec_begin = subl_database_iter->get_species_iterator();
		const auto spec_end = subl_database_iter->get_species_iterator_end();
		for (auto i = spec_begin; i != spec_end; ++i) ++speccount;
		if (!(speccount ==  1 && (*spec_begin) == "VA")) total_mixing_sites += (*subl_database_iter).stoi_coef;
		++subl_find;
		++subl_database_iter;
	}
	// We may have broken out of the loop before we finished summing up all the sites
	// This loop finishes the summation
	for (auto i = subl_database_iter; i != subl_database_iter_end; ++i) {
		int speccount = 0;
		total_sites += (*i).stoi_coef;
		const auto spec_begin = i->get_species_iterator();
		const auto spec_end = i->get_species_iterator_end();
		for (auto j = spec_begin; j != spec_end; ++j) ++speccount;
		if (!(speccount ==  1 && (*spec_begin) == "VA")) total_mixing_sites += (*i).stoi_coef;
	}
	if (subl_find == subl_end) {
		// we didn't find our sublattice
		BOOST_THROW_EXCEPTION(
				internal_error()
				<< str_errinfo(
						"Couldn't find sublattice in current phase"
				)
				<< specific_errinfo("Sublattice index out of bounds")
		);
	}
	result = result/total_mixing_sites; // normalize
	if (subl_find->at(specname) > 0) {
		// number of sites for this sublattice
		// + RT * num_sites/total_sites * (1 + ln(y(specindex,sublindex)))
		const double num_sites = (*subl_database_iter).stoi_coef;
		std::cout.precision(10);
		//std::cout << "y(" << specname << ") = " << subl_find->at(specname) << std::endl;
		result += SI_GAS_CONSTANT * conditions.statevars.at('T') * num_sites/total_mixing_sites * (1 + log(subl_find->at(specname)));
	}

	// TODO: add excess Gibbs energy term (dGex/dy is nonzero for R-K polynomials)
	result += 0;

	return result;
}
