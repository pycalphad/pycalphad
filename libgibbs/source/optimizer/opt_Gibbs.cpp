/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// opt_Gibbs.cpp -- definition for Gibbs energy optimizer

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/optimizer.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include "libgibbs/include/optimizer/halton.hpp"
#include "libtdb/include/logging.hpp"
#include "external/coin/IpTNLP.hpp"
#include <sstream>

using namespace Ipopt;

/* Constructor. */
GibbsOpt::GibbsOpt(
		const Database &DB,
		const evalconditions &sysstate) :
			conditions(sysstate),
			phase_iter(DB.get_phase_iterator()),
			phase_end(DB.get_phase_iterator_end())
{
	int varcount = 0;
	int activephases = 0;
	logger opt_log(journal::keywords::channel = "optimizer");

	for (auto i = DB.get_phase_iterator(); i != DB.get_phase_iterator_end(); ++i) {
		if (conditions.phases.find(i->first) != conditions.phases.end()) {
			if (conditions.phases.at(i->first) == PhaseStatus::ENTERED) phase_col[i->first] = i->second;
		}
	}



	phase_iter = phase_col.cbegin();
	phase_end = phase_col.cend();
	if (conditions.elements.cbegin() == conditions.elements.cend()) BOOST_LOG_SEV(opt_log, critical) << "Missing element conditions!";
	if (phase_iter == phase_end) BOOST_LOG_SEV(opt_log, critical) << "No phases found!";

	// build_variable_map() will fill main_indices
	main_ss = build_variable_map(phase_iter, phase_end, conditions, main_indices);

	// load the parameters from the database
	parameter_set pset = DB.get_parameter_set();

	// this is the part where we look up the models enabled for each phase and call their AST builders
	// then we build a master Gibbs AST for the objective function
	// TODO: right now the models called are hard-coded, in the future this will be typedef-dependent
	for (auto i = phase_iter; i != phase_end; ++i) {
		if (conditions.phases[i->first] != PhaseStatus::ENTERED) continue;
		++activephases;
		boost::spirit::utree phase_ast;
		boost::spirit::utree temptree;
		// build an AST for the given phase
		boost::spirit::utree curphaseref = PureCompoundEnergyModel(i->first, main_ss, pset).get_ast();
		BOOST_LOG_SEV(opt_log, debug) << i->first << "ref" << std::endl << curphaseref << std::endl;
		boost::spirit::utree idealmix = IdealMixingModel(i->first, main_ss).get_ast();
		BOOST_LOG_SEV(opt_log, debug) << i->first << "idmix" << std::endl << idealmix << std::endl;
		boost::spirit::utree redlichkister = RedlichKisterExcessEnergyModel(i->first, main_ss, pset).get_ast();
		BOOST_LOG_SEV(opt_log, debug) << i->first << "excess" << std::endl << redlichkister << std::endl;

		// sum the contributions
		phase_ast.push_back("+");
		phase_ast.push_back(idealmix);
		temptree.push_back("+");
		temptree.push_back(curphaseref);
		temptree.push_back(redlichkister);
		phase_ast.push_back(temptree);
		temptree.clear();

		// multiply by phase fraction variable
		temptree.push_back("*");
		temptree.push_back(i->first + "_FRAC");
		temptree.push_back(phase_ast);
		phase_ast.swap(temptree);
		temptree.clear();

		// add phase AST to master AST
		if (activephases != 1) {
			// this is not the only / first phase
			temptree.push_back("+");
			temptree.push_back(master_tree);
			temptree.push_back(phase_ast);
			master_tree.swap(temptree);
		}
		else master_tree.swap(phase_ast);

	}
	BOOST_LOG_SEV(opt_log, debug) << "master_tree: " << master_tree << std::endl;

	// Calculate first derivative ASTs of all variables
	for (auto i = main_indices.begin(); i != main_indices.end(); ++i) {
		first_derivatives[i->second] = differentiate_utree(master_tree, i->first);
	}


	// Add the mandatory constraints to the ConstraintManager
	if (activephases > 1)
		cm.addConstraint(
				PhaseFractionBalanceConstraint(
						phase_iter, phase_end
					)
				); // Add the mass balance constraint to ConstraintManager (mandatory)

	// Add the sublattice site fraction constraints (mandatory)
	for (auto i = phase_iter; i != phase_end; ++i) {
		if (conditions.phases[i->first] != PhaseStatus::ENTERED) continue;
		for (auto j = i->second.get_sublattice_iterator(); j != i->second.get_sublattice_iterator_end();++j) {
			std::vector<std::string> subl_list;
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					subl_list.push_back(*k); // Add to the list
				}
			}
			if (subl_list.size() == 1) {
				std::stringstream ss;
				ss << i->first << "_" << std::distance(i->second.get_sublattice_iterator(),j) << "_" << *(subl_list.begin());
				fixed_indices.push_back(main_indices[ss.str()]);
			}
			if (subl_list.size() > 1 ) {
				cm.addConstraint(
						SublatticeBalanceConstraint(
								i->first,
								std::distance(i->second.get_sublattice_iterator(),j),
								subl_list.cbegin(),
								subl_list.cend()
						)
				);
			}
		}
	}

	// Add any user-specified constraints to the ConstraintManager

	for (auto i = conditions.xfrac.cbegin(); i != conditions.xfrac.cend(); ++i) {
		cm.addConstraint(MassBalanceConstraint(phase_iter, phase_end, i->first, i->second));
	}

	for (auto i = cm.begin(); i != cm.end(); ++i) {
		BOOST_LOG_SEV(opt_log, debug) << i->name << " LHS: " << i->lhs << std::endl;
		BOOST_LOG_SEV(opt_log, debug) << i->name << " RHS: " << i->rhs << std::endl;
	}

	// Calculate first derivative ASTs of all constraints
	for (auto i = main_indices.cbegin(); i != main_indices.cend(); ++i) {
		// for each variable, calculate derivatives of all the constraints
		for (auto j = cm.begin(); j != cm.end(); ++j) {
			boost::spirit::utree lhs = differentiate_utree(j->lhs, i->first);
			boost::spirit::utree rhs = differentiate_utree(j->rhs, i->first);
			boost::spirit::utree subtract_tree;
			subtract_tree.push_back("-");
			subtract_tree.push_back(lhs);
			subtract_tree.push_back(rhs);
			jac_g_trees.push_back(subtract_tree);
		}
	}

	// Build the index map
	for (auto i = phase_iter; i != phase_end; ++i) {
		if (conditions.phases[i->first] != PhaseStatus::ENTERED) continue;
		//std::cout << "x[" << varcount << "] = " << i->first << " phasefrac" << std::endl;
		var_map.phasefrac_iters.push_back(boost::make_tuple(varcount,varcount+1,i));
		++varcount;
		for (auto j = i->second.get_sublattice_iterator(); j != i->second.get_sublattice_iterator_end();++j) {
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					// This site matches one of our elements under investigation
					// Add it to the list of sitefracs
					// +1 for a sitefraction
					//std::cout << "x[" << varcount << "] = (" << i->first << "," << std::distance(i->second.get_sublattice_iterator(),j) << "," << *k << ")" << std::endl;
					var_map.sitefrac_iters.resize(std::distance(phase_iter,i)+1);
					var_map.sitefrac_iters[std::distance(phase_iter,i)].resize(std::distance(i->second.get_sublattice_iterator(),j)+1);
					var_map.sitefrac_iters[std::distance(phase_iter,i)][std::distance(i->second.get_sublattice_iterator(),j)][*k] =
						std::make_pair(varcount,i);
					++varcount;
				}
			}
		}
	}
}

GibbsOpt::~GibbsOpt()
{}

bool GibbsOpt::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                         Index& nnz_h_lag, IndexStyleEnum& index_style)
{
  // number of variables
  n = main_indices.size();
  //std::cout << "n = " << n << std::endl;
  // one phase fraction balance constraint (for multi-phase)
  // plus all the sublattice fraction balance constraints
  // plus all the mass balance constraints


  m = std::distance(cm.begin(),cm.end());
  //std::cout << "m = " << m << std::endl;

  // nonzeros in the jacobian of the lagrangian
  /*if (phasecount > 1) {
	  nnz_jac_g = phasecount + (speccount-1)*phasecount + balanced_species_in_each_sublattice + balancedsitefraccount;
  }
  else {
	  // single-phase case
	  nnz_jac_g = (speccount-1)*phasecount + balanced_species_in_each_sublattice + balancedsitefraccount;
  }*/

  nnz_jac_g = n*m; // TODO: temporary fix until I get better at non-zero detection

  index_style = C_STYLE;
  return true;
}

bool GibbsOpt::get_bounds_info(Index n, Number* x_l, Number* x_u,
                            Index m_num, Number* g_l, Number* g_u)
{
	for (Index i = 0; i < n; ++i) {
		x_l[i] = 0;
		x_u[i] = 1;
	}

	Index cons_index = 0;
	auto sitefrac_begin = var_map.sitefrac_iters.begin();
	auto sitefrac_end = var_map.sitefrac_iters.end();
	if (std::distance(sitefrac_begin, sitefrac_end) == 1) {
		// single phase optimization, fix the value of the phase fraction at 1
		x_l[0] = x_u[0] = 1;
	}
	for (auto i = fixed_indices.begin(); i != fixed_indices.end(); ++i) {
		x_l[*i] = x_u[*i] = 1;
	}

	// Set bounds for constraints
	for (auto i = cm.begin(); i != cm.end(); ++i) {
		if (i->op == ConstraintOperatorType::EQUALITY_CONSTRAINT) {
			g_l[std::distance(cm.begin(),i)] = 0;
			g_u[std::distance(cm.begin(),i)] = 0;
		}
	}

	return true;
}

bool GibbsOpt::get_starting_point(Index n, bool init_x, Number* x,
	bool init_z, Number* z_L, Number* z_U,
	Index m, bool init_lambda,
	Number* lambda)
{
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
	return true;
}

bool GibbsOpt::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
	// return the value of the objective function
	//std::cout << "enter process_utree" << std::endl;
	obj_value = process_utree(master_tree, conditions, main_indices, (double*)x).get<double>();
	//std::cout << "exit process_utree" << std::endl;
	//std::cout << "eval_f: " << obj_value << " (new) == " << result << std::endl;
	return true;
}

bool GibbsOpt::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{
	// return the gradient of the objective function grad_{x} f(x)
	// calculate dF/dy(l,s,j)
	//std::cout << "eval_grad_f entered" << std::endl;
	for (auto i = first_derivatives.begin(); i != first_derivatives.end(); ++i) {
		grad_f[i->first] = process_utree(i->second, conditions, main_indices, (double*) x).get<double>();
	}

	//std::cout << "eval_grad_f exit" << std::endl;
	return true;
}

bool GibbsOpt::eval_g(Index n, const Number* x, bool new_x, Index m_num, Number* g)
{
	logger opt_log(journal::keywords::channel = "optimizer");
	BOOST_LOG_SEV(opt_log, debug) << "entering eval_g" << std::endl;
	// return the value of the constraints: g(x)
	const auto cons_begin = cm.begin();
	const auto cons_end = cm.end();

	for (auto i = cons_begin; i != cons_end; ++i) {
		// Calculate left-hand side and right-hand side of all constraints
		//BOOST_LOG_SEV(opt_log, debug) << "Constraint " << std::distance(cons_begin,i) << std::endl;
		//BOOST_LOG_SEV(opt_log, debug) << i->name << " LHS: " << i->lhs << std::endl;
		//BOOST_LOG_SEV(opt_log, debug) << i->name << " RHS: " << i->rhs << std::endl;
		double lhs = process_utree(i->lhs, conditions, main_indices, (double*)x).get<double>();
		//BOOST_LOG_SEV(opt_log, debug) << i->name << " LHS: " << lhs << std::endl;
		double rhs = process_utree(i->rhs, conditions, main_indices, (double*)x).get<double>();
		//BOOST_LOG_SEV(opt_log, debug) << i->name << " RHS: " << rhs << std::endl;
		g[std::distance(cons_begin,i)] = lhs - rhs;
	}

	BOOST_LOG_SEV(opt_log, debug) << "exiting eval_g" << std::endl;
  return true;
}

bool GibbsOpt::eval_jac_g(Index n, const Number* x, bool new_x,
	Index m_num, Index nele_jac, Index* iRow, Index *jCol,
	Number* values)
{
	Index jac_index = 0;
	logger opt_log(journal::keywords::channel = "optimizer");
	if (values == NULL) {
		BOOST_LOG_SEV(opt_log, debug) << "entering eval_jac_g values == NULL" << std::endl;
		for (auto i = main_indices.cbegin(); i != main_indices.cend(); ++i) {
			BOOST_LOG_SEV(opt_log, debug) << "Variable " << i->first << std::endl;
			// for each variable-constraint combination, give it a jac_index
			for (auto j = cm.begin(); j != cm.end(); ++j) {
				//BOOST_LOG_SEV(opt_log, debug) << "Constraint " << std::distance(cm.begin(),j) << std::endl;
				//BOOST_LOG_SEV(opt_log, debug) << "iRow[" << jac_index << "] = " << std::distance(cm.begin(),j) << std::endl;
				//BOOST_LOG_SEV(opt_log, debug) << "jCol[" << jac_index << "] = " << i->second << std::endl;
				iRow[jac_index] = std::distance(cm.begin(),j);
				jCol[jac_index] = i->second;
				++jac_index;
			}
		}
		BOOST_LOG_SEV(opt_log, debug) << "exit eval_jac_g without values" << std::endl;
	}
	else {
		for (auto i = jac_g_trees.cbegin(); i != jac_g_trees.cend(); ++i) {
			values[std::distance(jac_g_trees.cbegin(),i)] = process_utree(*i, conditions, main_indices, (double*)x).get<double>();
		}
		BOOST_LOG_SEV(opt_log, debug) << "exit eval_jac_g with values" << std::endl;
	}
	return true;
}

bool GibbsOpt::eval_h(Index n, const Number* x, bool new_x,
                   Number obj_factor, Index m, const Number* lambda,
                   bool new_lambda, Index nele_hess, Index* iRow,
                   Index* jCol, Number* values)
{
	// No explicit evaluation of the Hessian
	return false;

	Index h_idx = 0;
	if (values == NULL) {
		for (auto i = main_indices.cbegin(); i != main_indices.cend(); ++i) {
			// for each variable, calculate derivatives of all the constraints
			for (auto j = cm.begin(); j != cm.end(); ++j) {
				double lhs = differentiate_utree(j->lhs, conditions, i->first, main_indices, (double*) x).get<double>();
				double rhs = differentiate_utree(j->rhs, conditions, i->first, main_indices, (double*) x).get<double>();
				values[h_idx] = lhs - rhs;
				++h_idx;
			}
		}
	}
	else {
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
	auto sitefrac_begin = var_map.sitefrac_iters.begin();
	sitefracs thesitefracs;
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
					//std::cout << "y(" << cur_phase->first << "," << *k << ") = " << x[sitefracindex] << std::endl;
				}
			}
			subls_vec.push_back(std::make_pair((*j).stoi_coef,subl_map));
		}
		//thesitefracs.push_back(std::make_pair(cur_phase->first,subls_vec));
		ph_map[cur_phase->first] = std::make_pair(fL,subls_vec);
	}
}
