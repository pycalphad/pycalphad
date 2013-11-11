/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Model for magnetic ordering according to Inden-Hillert-Jarl (IHJ)
// Reference: Inden, 1976. Hillert and Jarl, 1978.
// Handling for anti-ferromagnetic (AFM) states done according to Hertzman and Sundman, 1982.
// Implementation follows from a review of the IHJ model by W. Xiong et al., 2012.
// Note: This is the IHJ model, not Xiong's model.

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include <string>
#include <sstream>
#include <set>
#include <limits>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/tuple/tuple.hpp>

using boost::spirit::utree;
typedef boost::spirit::utree_type utree_type;
using boost::multi_index_container;
using namespace boost::multi_index;

// Apply op with root_tree as left hand side and new_tree as right hand side
utree a_o (const utree &root_tree, const utree &new_tree, const std::string &op) {
	utree temp_tree;
	temp_tree.push_back(op);
	temp_tree.push_back(root_tree);
	temp_tree.push_back(new_tree);
	return temp_tree;
}
// Apply op with root_tree as left hand side
utree a_o (const utree &root_tree, const std::string &op) {
	utree temp_tree;
	temp_tree.push_back(op);
	temp_tree.push_back(root_tree);
	return temp_tree;
}

utree max_magnetic_entropy(const utree &beta);
utree magnetic_polynomial(const utree &tc_tree, const double &p);

IHJMagneticModel::IHJMagneticModel(
		const std::string &phasename,
		const sublattice_set &subl_set,
		const parameter_set &param_set,
		const double &afm_factor,
		const double &sro_enthalpy_order_fraction
		) : EnergyModel(phasename, subl_set, param_set) {
	if (afm_factor == 0 || sro_enthalpy_order_fraction == 0) {
		// There is no magnetic contribution
		model_ast = utree(0);
		return;
	}
	sublattice_set_view ssv;
	parameter_set_view psv;
	parameter_set_view psv_subview_tc, psv_subview_bm;
	utree Curie_temperature, mean_magnetic_moment;
	std::string scantype;
	boost::multi_index::index<sublattice_set,phases>::type::iterator ic0,ic1;
	boost::multi_index::index<parameter_set_view,type_index>::type::iterator it0, it1;
	boost::multi_index::index<parameter_set,phase_index>::type::iterator pa_start,pa_end;

	// Get all the sublattices for this phase
	boost::tuples::tie(ic0,ic1)=get<phases>(subl_set).equal_range(phasename);
	// Construct a view from the iterators
	while (ic0 != ic1) {
		ssv.insert(&*ic0);
		++ic0;
	}
	// Get all the parameters for this phase
	boost::tuples::tie(pa_start,pa_end)=get<phase_index>(param_set).equal_range(phasename);
	// Construct a view from the iterators
	while (pa_start != pa_end) {
		psv.insert(&*pa_start);
		++pa_start;
	}

	// build subview to the parameter "TC"
	scantype = "TC";
	boost::tuples::tie(it0,it1)=get<type_index>(psv).equal_range(scantype);

	// Construct a subview from the view
	while (it0 != it1) {
		psv_subview_tc.insert(*it0);
		++it0;
	}
	Curie_temperature = permute_site_fractions(ssv, sublattice_set_view(), psv_subview_tc, (int)0);
	Curie_temperature = a_o(Curie_temperature,
			permute_site_fractions_with_interactions(ssv, sublattice_set_view(), psv_subview_tc, (int)0),
			"+"
	);
	Curie_temperature = a_o(Curie_temperature, afm_factor, "/"); // divide TC by the AFM factor
	std::cout << "Curie_temperature: " << Curie_temperature << std::endl;
	Curie_temperature = simplify_utree(Curie_temperature);
	std::cout << "Curie temperature after process_utree: " << Curie_temperature << std::endl;
	if (is_zero_tree(Curie_temperature)) {
		// Transition temperature is always zero, no magnetic contribution
		model_ast = utree(0);
		return;
	}

	// Now find parameters of type "BMAGN"
	scantype = "BMAGN";
	boost::tuples::tie(it0,it1)=get<type_index>(psv).equal_range(scantype);

	// Construct a subview from the view
	while (it0 != it1) {
		psv_subview_bm.insert(*it0);
		++it0;
	}
	mean_magnetic_moment = permute_site_fractions(ssv, sublattice_set_view(), psv_subview_bm, (int)0);
	mean_magnetic_moment = a_o(mean_magnetic_moment,
			permute_site_fractions_with_interactions(ssv, sublattice_set_view(), psv_subview_bm, (int)0),
			"+"
	);
	mean_magnetic_moment = a_o(mean_magnetic_moment, afm_factor, "/"); // divide BMAGN by the AFM factor
	mean_magnetic_moment = simplify_utree(mean_magnetic_moment);
	if (is_zero_tree(mean_magnetic_moment)) {
		// Mean magnetic moment is always zero, no magnetic contribution
		model_ast = utree(0);
		return;
	}

	model_ast = a_o("T", max_magnetic_entropy(mean_magnetic_moment), "*");
	model_ast = a_o(model_ast, magnetic_polynomial(Curie_temperature, sro_enthalpy_order_fraction), "*");


	normalize_utree(model_ast, ssv);
}

utree magnetic_polynomial(const utree &tc_tree, const double &p) {
	// These are constant factors from the heat capacity integration
	double A = (518.0/1125.0) + ((11692.0/15975.0)*((1.0/p) - 1.0));
	double B = 79.0/(140*p);
	double C = (474.0/497.0)*((1.0/p)-1.0);
	utree ret_tree, tau, subcritical_tree, supercritical_tree;

	tau = a_o("T", tc_tree, "/");

	// TODO: This is a mess. Using the utree visitation interface might make this better.

	// First calculate the polynomial for tau < 1
	utree taum1 = a_o(B, a_o(tau, -1, "**"), "*");
	utree tau3 = a_o(1.0/6.0, a_o(tau, 3, "**"), "*");
	utree tau9 = a_o(1.0/135.0, a_o(tau, 9, "**"), "*");
	utree tau15 = a_o(1.0/600.0, a_o(tau, 15, "**"), "*");
	utree total_taus = a_o(C, a_o(a_o(tau3, tau9, "+"), tau15, "+"), "*");
	total_taus = a_o(taum1, total_taus, "+");
	total_taus = a_o(-1.0/A, total_taus, "*");
	subcritical_tree = a_o(1, total_taus, "-");

	// Now calculate the polynomial for tau >= 1
	utree taum5 = a_o(1.0/10.0, a_o(tau, -5, "**"), "*");
	utree taum15 = a_o(1.0/315.0, a_o(tau, -15, "**"), "*");
	utree taum25 = a_o(1.0/1500.0, a_o(tau, -25, "**"), "*");
	total_taus = a_o(a_o(taum5, taum15, "+"), taum25, "+");
	supercritical_tree = a_o(-1/A, total_taus, "*");

	ret_tree.push_back("@");
	ret_tree.push_back(tau);
	ret_tree.push_back(-std::numeric_limits<double>::max());
	ret_tree.push_back(1);
	ret_tree.push_back(subcritical_tree);
	ret_tree.push_back("@");
	ret_tree.push_back(tau);
	ret_tree.push_back(1);
	ret_tree.push_back(std::numeric_limits<double>::max());
	ret_tree.push_back(supercritical_tree);

	return ret_tree;
}

utree max_magnetic_entropy(const utree &beta) {
	// beta is the mean magnetic moment
	// The return value is the maximum magnetic entropy of an element undergoing the FM disordering transition
	utree ret_tree;
	ret_tree = a_o(SI_GAS_CONSTANT, a_o(a_o(beta, 1, "+"), "LN"), "*"); // R*ln(beta + 1)

	return ret_tree;
}
