/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// opt_Gibbs.cpp -- definition for Gibbs energy optimizer

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include "libgibbs/include/optimizer/halton.hpp"
#include "libtdb/include/logging.hpp"
#include "external/coin/IpTNLP.hpp"
#include <sstream>

using namespace Ipopt;

bool GibbsOpt::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
		Index& nnz_h_lag, IndexStyleEnum& index_style)
{
	BOOST_LOG_NAMED_SCOPE("GibbsOpt::get_nlp_info");
	BOOST_LOG_SEV(opto_log, debug) << "entering get_nlp_info";
	// number of variables
	n = main_indices.size();
	BOOST_LOG_SEV(opto_log, debug) << "Number of variables: " << n;
	// number of constraints
	m = std::distance(cm.constraints.begin(),cm.constraints.end());
	BOOST_LOG_SEV(opto_log, debug) << "Number of constraints: " << m;
	// number of nonzeros in Jacobian of constraints
	nnz_jac_g = jac_g_trees.size();
	BOOST_LOG_SEV(opto_log, debug) << "Number of nonzeros in Jacobian of constraints: " << nnz_jac_g;
	// number of nonzeros in the Hessian
	nnz_h_lag = hessian_data.size();
	BOOST_LOG_SEV(opto_log, debug) << "Number of nonzeros in Hessian: " << nnz_h_lag;
	// indices start at 0
	index_style = C_STYLE;
	BOOST_LOG_SEV(opto_log, debug) << "exiting get_nlp_info";
	return true;
}

bool GibbsOpt::get_bounds_info(Index n, Number* x_l, Number* x_u,
                            Index m_num, Number* g_l, Number* g_u)
{
	BOOST_LOG_NAMED_SCOPE("GibbsOpt::get_bounds_info");
	BOOST_LOG_SEV(opto_log, debug) << "entering get_bounds_info";
	BOOST_LOG_SEV(opto_log, debug) << "m_num: " << m_num;
	if (cm.constraints.size() > m_num)
		BOOST_THROW_EXCEPTION(bounds_error() << str_errinfo("Specified fixed constraint variable index is out of bounds"));
	for (Index i = 0; i < n; ++i) {
		BOOST_LOG_SEV(opto_log, debug) << "Setting main variable bounds for x = " << i;
		x_l[i] = 0;
		x_u[i] = 1;
	}
	BOOST_LOG_SEV(opto_log, debug) << "All main variable bounds set";

	for (auto i = fixed_indices.begin(); i != fixed_indices.end(); ++i) {
		if (*i >= n) BOOST_THROW_EXCEPTION(bounds_error() << str_errinfo("Specified fixed variable index is out of bounds"));
		BOOST_LOG_SEV(opto_log, debug) << "Fixing x[" << *i << "] = 1";
		x_l[*i] = 1;
		x_u[*i] = 1;
	}
	BOOST_LOG_SEV(opto_log, debug) << "Completed fixed index optimization check";
	// Set bounds for constraints
	for (auto i = cm.constraints.begin(); i != cm.constraints.end(); ++i) {
		if (i->op == ConstraintOperatorType::EQUALITY_CONSTRAINT) {
			BOOST_LOG_SEV(opto_log, debug) << "Setting bounds for constraint " << std::distance(cm.constraints.begin(),i);
			g_l[std::distance(cm.constraints.begin(),i)] = 0;
			g_u[std::distance(cm.constraints.begin(),i)] = 0;
		}
	}
	BOOST_LOG_SEV(opto_log, debug) << "exiting get_bounds_info";
	return true;
}

bool GibbsOpt::get_starting_point(Index n, bool init_x, Number* x,
	bool init_z, Number* z_L, Number* z_U,
	Index m, bool init_lambda,
	Number* lambda)
{
	BOOST_LOG_NAMED_SCOPE("GibbsOpt::get_starting_point");
	BOOST_LOG_SEV(opto_log, debug) << "entering get_starting_point";
	const double numphases = var_map.phasefrac_iters.size();
	auto sitefrac_begin = var_map.sitefrac_iters.begin();
	double result = 0;
	int varcount = 0;
	// all phases
	for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
		const int phaseindex = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<0>();
		const Phase_Collection::const_iterator cur_phase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();
		x[phaseindex] = 1 / numphases; // phase fraction
		//std::cout << "x[" << phaseindex << "] = " << x[phaseindex] << std::endl;
		++varcount;
		for (auto j = cur_phase->second.get_sublattice_iterator(); j != cur_phase->second.get_sublattice_iterator_end();++j) {
			double speccount = 0;
			// Iterating through the sublattice twice is not very efficient,
			// but we only set the starting values once and this is far simpler to read
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					speccount = speccount + 1;
				}
			}
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					int sitefracindex = var_map.sitefrac_iters[std::distance(sitefrac_begin,i)][std::distance(cur_phase->second.get_sublattice_iterator(),j)][*k].first;
					x[sitefracindex] = 1 / speccount;
					//std::cout << "x[" << sitefracindex << "] = " << x[sitefracindex] << std::endl;
					++varcount;
				}
			}
		}
	}
	assert(varcount == m);
	BOOST_LOG_SEV(opto_log, debug) << "exiting get_starting_point";
	return true;
}

bool GibbsOpt::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
	BOOST_LOG_NAMED_SCOPE("GibbsOpt::eval_f");
	BOOST_LOG_SEV(opto_log, debug) << "entering eval_f";
	// return the value of the objective function
	try {
		BOOST_LOG_SEV(opto_log, debug) << "trying to evaluate master tree";
		double objective = 0;
		for (auto i = comp_sets.cbegin(); i != comp_sets.cend(); ++i){
			// multiply each composition set's objective function by its fraction; then add to the overall objective
			objective += x[main_indices[(*i)->name() + "_FRAC"]] * (*i)->evaluate_objective(conditions, main_indices,(double*)x);
		}
		obj_value = objective;
	}
	catch (boost::exception &e) {
		std::string specific_info, err_msg; // error message strings
		if (std::string const * mi = boost::get_error_info<specific_errinfo>(e) ) {
			specific_info = *mi;
		}
		if (std::string const * mi = boost::get_error_info<str_errinfo>(e) ) {
			err_msg = *mi;
		}
		BOOST_LOG_SEV(opto_log, critical) << "Exception: " << err_msg;
		BOOST_LOG_SEV(opto_log, critical) << "Reason: " << specific_info;
		BOOST_LOG_SEV(opto_log, critical) << boost::diagnostic_information(e);
		throw;
	}
	catch (std::exception &e) {
		BOOST_LOG_SEV(opto_log, critical) << "Exception: " << e.what();
		throw;
	}
	BOOST_LOG_SEV(opto_log, debug) << "exiting eval_f";
	return true;
}

bool GibbsOpt::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{
	BOOST_LOG_NAMED_SCOPE("GibbsOpt::eval_grad_f");
	BOOST_LOG_SEV(opto_log, debug) << "entering eval_grad_f";
	// initialize gradient to zero
	for (auto i = 0; i < n; ++i) {
		grad_f[i] = 0;
	}
	try {
		// For all composition sets, evaluate the gradient
		for (auto i = comp_sets.cbegin(); i != comp_sets.cend(); ++i){
			const std::map<int,double> gradmap = (*i)->evaluate_objective_gradient(conditions, main_indices, (double*)x);
			for (auto j = gradmap.cbegin(); j != gradmap.cend(); ++j) {
				grad_f[j->first] += j->second;
			}
		}
	}
	catch (boost::exception &e) {
		std::string specific_info, err_msg; // error message strings
		if (std::string const * mi = boost::get_error_info<specific_errinfo>(e) ) {
			specific_info = *mi;
		}
		if (std::string const * mi = boost::get_error_info<str_errinfo>(e) ) {
			err_msg = *mi;
		}
		BOOST_LOG_SEV(opto_log, critical) << "Exception: " << err_msg;
		BOOST_LOG_SEV(opto_log, critical) << "Reason: " << specific_info;
		BOOST_LOG_SEV(opto_log, critical) << boost::diagnostic_information(e);
		throw;
	}
	BOOST_LOG_SEV(opto_log, debug) << "exiting eval_grad_f";
	return true;
}

bool GibbsOpt::eval_g(Index n, const Number* x, bool new_x, Index m_num, Number* g)
{
	BOOST_LOG_NAMED_SCOPE("GibbsOpt::eval_g");
	BOOST_LOG_SEV(opto_log, debug) << "entering eval_g";
	// return the value of the constraints: g(x)
	const auto cons_begin = cm.constraints.begin();
	const auto cons_end = cm.constraints.end();
	try {
		for (auto i = cons_begin; i != cons_end; ++i) {
			// Calculate left-hand side and right-hand side of all constraints
			//BOOST_LOG_SEV(opto_log, debug) << "Constraint " << std::distance(cons_begin,i) << std::endl;
			//BOOST_LOG_SEV(opto_log, debug) << i->name << " LHS: " << i->lhs << std::endl;
			//BOOST_LOG_SEV(opto_log, debug) << i->name << " RHS: " << i->rhs << std::endl;
			double lhs = process_utree(i->lhs, conditions, main_indices, (double*)x).get<double>();
			//BOOST_LOG_SEV(opto_log, debug) << i->name << " LHS: " << lhs << std::endl;
			double rhs = process_utree(i->rhs, conditions, main_indices, (double*)x).get<double>();
			//BOOST_LOG_SEV(opto_log, debug) << i->name << " RHS: " << rhs << std::endl;
			g[std::distance(cons_begin,i)] = lhs - rhs;
		}
	}
	catch (boost::exception &e) {
		std::string specific_info, err_msg; // error message strings
		if (std::string const * mi = boost::get_error_info<specific_errinfo>(e) ) {
			specific_info = *mi;
		}
		if (std::string const * mi = boost::get_error_info<str_errinfo>(e) ) {
			err_msg = *mi;
		}
		BOOST_LOG_SEV(opto_log, critical) << "Exception: " << err_msg;
		BOOST_LOG_SEV(opto_log, critical) << "Reason: " << specific_info;
		BOOST_LOG_SEV(opto_log, critical) << boost::diagnostic_information(e);
		throw;
	}

	BOOST_LOG_SEV(opto_log, debug) << "exiting eval_g";
  return true;
}

bool GibbsOpt::eval_jac_g(Index n, const Number* x, bool new_x,
	Index m_num, Index nele_jac, Index* iRow, Index *jCol,
	Number* values)
{
	Index jac_index = 0;
	BOOST_LOG_NAMED_SCOPE("GibbsOpt::eval_jac_g");
	if (values == NULL) {
		BOOST_LOG_SEV(opto_log, debug) << "entering eval_jac_g values == NULL";
		for (auto i = jac_g_trees.cbegin(); i != jac_g_trees.cend(); ++i) {
			iRow[jac_index] = i->cons_index;
			jCol[jac_index] = i->var_index;
			++jac_index;
		}
		BOOST_LOG_SEV(opto_log, debug) << "exit eval_jac_g without values";
	}
	else {
		try {
		for (auto i = jac_g_trees.cbegin(); i != jac_g_trees.cend(); ++i) {
			values[std::distance(jac_g_trees.cbegin(),i)] = process_utree(i->ast, conditions, main_indices, (double*)x).get<double>();
		}
		}
		catch (boost::exception &e) {
			std::string specific_info, err_msg; // error message strings
			if (std::string const * mi = boost::get_error_info<specific_errinfo>(e) ) {
				specific_info = *mi;
			}
			if (std::string const * mi = boost::get_error_info<str_errinfo>(e) ) {
				err_msg = *mi;
			}
			BOOST_LOG_SEV(opto_log, critical) << "Exception: " << err_msg;
			BOOST_LOG_SEV(opto_log, critical) << "Reason: " << specific_info;
			BOOST_LOG_SEV(opto_log, critical) << boost::diagnostic_information(e);
			throw;
		}
		BOOST_LOG_SEV(opto_log, debug) << "exit eval_jac_g with values";
	}
	return true;
}

bool GibbsOpt::eval_h(Index n, const Number* x, bool new_x,
                   Number obj_factor, Index m, const Number* lambda,
                   bool new_lambda, Index nele_hess, Index* iRow,
                   Index* jCol, Number* values)
{
	Index h_idx = 0;
	BOOST_LOG_NAMED_SCOPE("GibbsOpt::eval_h");

	if (values == NULL) {
		BOOST_LOG_SEV(opto_log, debug) << "enter eval_h without values";
		for (auto i = hessian_data.cbegin(); i != hessian_data.cend(); ++i) {
			int varindex1 = i->var_index1;
			int varindex2 = i->var_index2;
			iRow[h_idx] = varindex1;
			jCol[h_idx] = varindex2;
			++h_idx;
		}
		BOOST_LOG_SEV(opto_log, debug) << "exit eval_h without values";
	}
	else {
		BOOST_LOG_SEV(opto_log, debug) << "enter eval_h with values";
		try {
			for (auto i = hessian_data.cbegin(); i != hessian_data.cend(); ++i) {
				int varindex1 = i->var_index1;
				int varindex2 = i->var_index2;
				values[h_idx] = 0; // initialize
				for (auto j = i->asts.cbegin(); j != i->asts.cend(); ++j) {
					BOOST_LOG_SEV(opto_log, debug) << "Hessian evaluation for constraint " << j->first << " (" << varindex1 << "," << varindex2 << ")";
					boost::spirit::utree hess_tree = process_utree(j->second, conditions, main_indices, (double*)x).get<double>();
					if (j->first == -1) {
						// objective portion
						values[h_idx] += obj_factor * hess_tree.get<double>();
					}
					else {
						// constraint portion
						values[h_idx] += lambda[j->first] * hess_tree.get<double>();
					}
				}
				++h_idx;
			}
		}
		catch (boost::exception &e) {
			std::string specific_info, err_msg; // error message strings
			if (std::string const * mi = boost::get_error_info<specific_errinfo>(e) ) {
				specific_info = *mi;
			}
			if (std::string const * mi = boost::get_error_info<str_errinfo>(e) ) {
				err_msg = *mi;
			}
			BOOST_LOG_SEV(opto_log, critical) << "Exception: " << err_msg;
			BOOST_LOG_SEV(opto_log, critical) << "Reason: " << specific_info;
			BOOST_LOG_SEV(opto_log, critical) << boost::diagnostic_information(e);
			throw;
		}
		BOOST_LOG_SEV(opto_log, debug) << "exit eval_h with values";
	}
	return true;
}

void GibbsOpt::finalize_solution(SolverReturn status,
                              Index n, const Number* x, const Number* z_L, const Number* z_U,
                              Index m_num, const Number* g, const Number* lambda,
                              Number obj_value,
			      const IpoptData* ip_data,
			      IpoptCalculatedQuantities* ip_cq)
{
	BOOST_LOG_NAMED_SCOPE("GibbsOpt::finalize_solution");
	BOOST_LOG_SEV(opto_log, debug) << "enter finalize_solution";
	auto sitefrac_begin = var_map.sitefrac_iters.begin();
	for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
		const Index phaseindex = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<0>();
		const Phase_Collection::const_iterator cur_phase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();
		const double fL = x[phaseindex]; // phase fraction

		constitution subls_vec;
		for (auto j = cur_phase->second.get_sublattice_iterator(); j != cur_phase->second.get_sublattice_iterator_end();++j) {
			std::unordered_map<std::string,double> subl_map;
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				Index sitefracindex;
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					sitefracindex = var_map.sitefrac_iters[std::distance(sitefrac_begin,i)][std::distance(cur_phase->second.get_sublattice_iterator(),j)][*k].first;
					subl_map[*k] = x[sitefracindex];
				}
			}
			subls_vec.push_back(std::make_pair((*j).stoi_coef,subl_map));
		}
		ph_map[cur_phase->first] = std::make_pair(fL,subls_vec);
	}
	// Test code for chemical potential calculation
	for (auto i = 0; i < m_num; ++i) {
			double mu = lambda[i];
			BOOST_LOG_SEV(opto_log, debug) << "MU(" << i << ") = " << mu;
	}
	BOOST_LOG_SEV(opto_log, debug) << "exit finalize_solution";
}
