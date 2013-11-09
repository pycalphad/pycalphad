/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition for GibbsOpt constructor and destructor

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include "libtdb/include/logging.hpp"
#include <sstream>

// Add new_tree to root_tree
void add_trees (boost::spirit::utree &root_tree, const boost::spirit::utree &new_tree) {
	boost::spirit::utree temp_tree;
	temp_tree.push_back("+");
	temp_tree.push_back(root_tree);
	temp_tree.push_back(new_tree);
	root_tree.swap(temp_tree);
}

GibbsOpt::GibbsOpt(
		const Database &DB,
		const evalconditions &sysstate) :
			conditions(sysstate),
			phase_iter(DB.get_phase_iterator()),
			phase_end(DB.get_phase_iterator_end())
{
	BOOST_LOG_NAMED_SCOPE("GibbsOpt::GibbsOpt");
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

	for (auto i = main_indices.begin(); i != main_indices.end(); ++i) {
		BOOST_LOG_SEV(opt_log, debug) << "Variable " << i->second << ": " << i->first;
	}

	// load the parameters from the database
	parameter_set pset = DB.get_parameter_set();

	// this is the part where we look up the models enabled for each phase and call their AST builders
	// then we build a master Gibbs AST for the objective function
	for (auto i = phase_iter; i != phase_end; ++i) {
		if (conditions.phases[i->first] != PhaseStatus::ENTERED) continue;
		++activephases;
		BOOST_LOG_SEV(opt_log, debug) << i->first << " magnetic_afm_factor: " << i->second.magnetic_afm_factor;
		BOOST_LOG_SEV(opt_log, debug) << i->first << " magnetic_sro_enthalpy_order_fraction: " << i->second.magnetic_sro_enthalpy_order_fraction;
		boost::spirit::utree phase_ast;
		boost::spirit::utree temptree;
		// build an AST for the given phase
		boost::spirit::utree curphaseref = PureCompoundEnergyModel(i->first, main_ss, pset).get_ast();
		BOOST_LOG_SEV(opt_log, debug) << i->first << " ref " << std::endl << curphaseref << std::endl;
		boost::spirit::utree idealmix = IdealMixingModel(i->first, main_ss).get_ast();
		BOOST_LOG_SEV(opt_log, debug) << i->first << " idmix " << std::endl << idealmix << std::endl;
		boost::spirit::utree redlichkister = RedlichKisterExcessEnergyModel(i->first, main_ss, pset).get_ast();
		BOOST_LOG_SEV(opt_log, debug) << i->first << " excess " << std::endl << redlichkister << std::endl;
		boost::spirit::utree magnetism =
				IHJMagneticModel(i->first, main_ss, pset,
						i->second.magnetic_afm_factor, i->second.magnetic_sro_enthalpy_order_fraction).get_ast();
		BOOST_LOG_SEV(opt_log, debug) << i->first << " magnetic " << std::endl << magnetism << std::endl;

		// sum the contributions
		add_trees(phase_ast, idealmix);
		add_trees(phase_ast, curphaseref);
		add_trees(phase_ast, redlichkister);
		add_trees(phase_ast, magnetism);

		// multiply by phase fraction variable
		temptree.push_back("*");
		temptree.push_back(i->first + "_FRAC");
		temptree.push_back(phase_ast);
		phase_ast.swap(temptree);
		temptree.clear();

		// add phase AST to master AST
		if (activephases != 1) {
			// this is not the only / first phase
			add_trees(master_tree, phase_ast);
		}
		else master_tree.swap(phase_ast);

	}
	//BOOST_LOG_SEV(opt_log, debug) << "master_tree: " << master_tree << std::endl;


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
	for (auto i = cm.begin() ; i != cm.end(); ++i) {
		BOOST_LOG_SEV(opt_log, debug) << "Constraint " << i->name << " LHS: " << i->lhs;
		BOOST_LOG_SEV(opt_log, debug) << "Constraint " << i->name << " RHS: " << i->rhs;
	}

	// Calculate first derivative ASTs of all variables
	for (auto i = main_indices.begin(); i != main_indices.end(); ++i) {
		first_derivatives[i->second] = differentiate_utree(master_tree, i->first);
		//BOOST_LOG_SEV(opt_log, debug) << "First derivative w.r.t. " << i->first << "(" << i->second << ") = " << first_derivatives[i->second] << std::endl;
	}

	// Calculate first derivative ASTs of all constraints
	for (auto i = main_indices.cbegin(); i != main_indices.cend(); ++i) {
		// for each variable, calculate derivatives of all the constraints
		for (auto j = cm.begin(); j != cm.end(); ++j) {
			boost::spirit::utree lhs = differentiate_utree(j->lhs, i->first);
			boost::spirit::utree rhs = differentiate_utree(j->rhs, i->first);
			boost::spirit::utree lhstest = process_utree(lhs);
			boost::spirit::utree rhstest = process_utree(rhs);
			if (lhstest.which() == boost::spirit::utree_type::double_type && rhstest.which() == boost::spirit::utree_type::double_type) {
				double lhsget, rhsget;
				lhsget = lhstest.get<double>();
				rhsget = rhstest.get<double>();
				if (lhsget == rhsget) continue; // don't add zeros to the Jacobian
			}
			boost::spirit::utree subtract_tree;
			subtract_tree.push_back("-");
			subtract_tree.push_back(lhs);
			subtract_tree.push_back(rhs);
			int var_index = i->second;
			int cons_index = std::distance(cm.begin(),j);
			jac_g_trees.push_back(jacobian_entry(cons_index,var_index,false,subtract_tree));
			BOOST_LOG_SEV(opt_log, debug) << "Jacobian of constraint  " << cons_index << " wrt variable " << var_index << " pre-calculated";
		}
	}

	// Calculate second derivatives of objective function and constraints
	for (auto i = main_indices.cbegin(); i != main_indices.cend(); ++i) {
		// objective portion
		for (auto j = main_indices.cbegin(); j != main_indices.cend(); ++j) {
			// second derivative of obj function w.r.t i,j
			if (i->second > j->second) continue; // skip upper triangular
			boost::spirit::utree obj_second_deriv = differentiate_utree(first_derivatives[i->second], j->first);
			boost::spirit::utree test_tree = process_utree(obj_second_deriv);
			hessian_set::iterator h_iter, h_end;
			// don't add zeros to the Hessian
			// TODO: this misses some of the zeros
			if (test_tree.which() == boost::spirit::utree_type::double_type && test_tree.get<double>() == 0) continue;

			h_iter = hessian_data.lower_bound(boost::make_tuple(i->second,j->second));
			h_end = hessian_data.upper_bound(boost::make_tuple(i->second,j->second));
			// create a new Hessian record if it does not exist
			if (h_iter == h_end) h_iter = hessian_data.insert(hessian_entry(i->second,j->second)).first;
			hessian_entry h_entry = *h_iter;
			h_entry.asts[-1] = obj_second_deriv; // set AST for objective Hessian
			hessian_data.replace(h_iter, h_entry); // update original entry

			BOOST_LOG_SEV(opt_log, debug) << "Hessian of objective  ("
					<< i->second << "," << j->second << ") pre-calculated";
		}

		// each of the constraints
		// for each variable, calculate derivatives of the Jacobian w.r.t all the constraints
		for (auto j = jac_g_trees.cbegin(); j != jac_g_trees.cend(); ++j) {
			if (i->second > j->var_index) continue; // skip upper triangular
			// second derivative of constraint jac_g_trees->cons_index w.r.t jac_g_trees->var_index, i->second
			boost::spirit::utree cons_second_deriv = differentiate_utree(j->ast, i->first);
			boost::spirit::utree test_tree = process_utree(cons_second_deriv);
			hessian_set::iterator h_iter, h_end;
			// don't add zeros to the Hessian
			if (test_tree.which() == boost::spirit::utree_type::double_type && test_tree.get<double>() == 0) continue;

			h_iter = hessian_data.lower_bound(boost::make_tuple(i->second,j->var_index));
			h_end = hessian_data.upper_bound(boost::make_tuple(i->second,j->var_index));
			// create a new Hessian record if it does not exist
			if (h_iter == h_end) h_iter = hessian_data.insert(hessian_entry(i->second,j->var_index)).first;
			hessian_entry h_entry = *h_iter;
			h_entry.asts[j->cons_index] = cons_second_deriv; // set AST for constraint Hessian
			hessian_data.replace(h_iter, h_entry); // update original entry
			BOOST_LOG_SEV(opt_log, debug) << "Hessian of constraint  "
					<< j->cons_index << " (" << j->var_index << "," << i->second << ") pre-calculated";
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
