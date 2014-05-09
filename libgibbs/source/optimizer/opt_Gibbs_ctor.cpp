/*=============================================================================
	Copyright (c) 2012-2014 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition for GibbsOpt constructor and destructor

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/utils/ast_container_rename.hpp"
#include "libgibbs/include/utils/math_expr.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include "libgibbs/include/optimizer/utils/build_variable_map.hpp"
#include "libgibbs/include/optimizer/utils/convex_hull.hpp"
#include "libgibbs/include/optimizer/utils/ezd_minimization.hpp"
#include "libgibbs/include/optimizer/global_minimization.hpp"
#include "libtdb/include/logging.hpp"
#include <sstream>

using namespace Optimizer;

// Add new_tree to root_tree
void add_trees ( boost::spirit::utree &root_tree, const boost::spirit::utree &new_tree )
{
    boost::spirit::utree temp_tree;
    temp_tree.push_back ( "+" );
    temp_tree.push_back ( root_tree );
    temp_tree.push_back ( new_tree );
    root_tree.swap ( temp_tree );
}

GibbsOpt::GibbsOpt (
    const Database &DB,
    const evalconditions &sysstate ) :
    conditions ( sysstate )
{
    BOOST_LOG_NAMED_SCOPE ( "GibbsOpt::GibbsOpt" );
    BOOST_LOG_CHANNEL_SEV ( opto_log, "optimizer", debug ) << "enter ctor";
    auto varcount = 0;
    auto activephases = 0;
    parameter_set pset;
    Phase_Collection phase_col;

    for ( auto i = DB.get_phase_iterator(); i != DB.get_phase_iterator_end(); ++i ) {
        if ( conditions.phases.find ( i->first ) != conditions.phases.end() ) {
            if ( conditions.phases.at ( i->first ) == PhaseStatus::ENTERED ) {
                phase_col[i->first] = i->second;
            }
        }
    }

    if ( conditions.elements.cbegin() == conditions.elements.cend() ) {
        BOOST_LOG_SEV ( opto_log, critical ) << "No components entered!";
    }
    if ( phase_col.begin() == phase_col.end() ) {
        BOOST_LOG_SEV ( opto_log, critical ) << "No phases found!";
    }

    // build_variable_map() will fill main_indices by reference
    // main_indices is used during the optimization as a simplified variable map
    main_ss = build_variable_map ( phase_col.begin(), phase_col.end(), conditions, main_indices );

    for ( auto i = main_indices.left.begin(); i != main_indices.left.end(); ++i ) {
        BOOST_LOG_SEV ( opto_log, debug ) << "Variable " << i->second << ": " << i->first;
    }

    // load the parameters from the database
    pset = DB.get_parameter_set();

    // this is the part where we look up the models enabled for each phase and call their AST builders
    // then we build a master Gibbs AST for the objective function
    auto temp_phase_col = phase_col; // We modify phase_col, so we should be careful here
    for ( auto i = temp_phase_col.begin(); i != temp_phase_col.end(); ++i ) {
        comp_sets.emplace ( i->first, CompositionSet ( i->second, pset, main_ss, main_indices ) );
    }
    // TODO: Move these changeable parameters somewhere
    const double subdivisions_per_axis = 10;
    const std::size_t critical_edge_length = 0.01;
    
    // Rebind functions to use user-defined parameters
    GlobalMinimizer<double,double>::PointSampleFunctor PointSampleFunction = std::bind ( details::AdaptiveSimplexSample, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, subdivisions_per_axis );
    GlobalMinimizer<double,double>::InternalHullFunctor InternalHullFunction = std::bind ( details::internal_lower_convex_hull, std::placeholders::_1, std::placeholders::_2, critical_edge_length, std::placeholders::_3 );

    // GlobalMinimizer will modify comp_sets and set the starting points automatically
    GlobalMinimizer<double,double> grid ( comp_sets, main_ss, conditions, PointSampleFunction, InternalHullFunction );
    
    // Rebuild the index map now that phases have been renamed and removed
    BOOST_LOG_SEV ( opto_log, debug ) << "Rebuilding variable map";
    main_indices.left.clear();
    main_ss = build_variable_map ( phase_col.begin(), phase_col.end(), conditions, main_indices );
    
    activephases = comp_sets.size();

    // Add the mandatory constraints to the ConstraintManager
    if ( activephases > 1 )
        cm.addConstraint (
            PhaseFractionBalanceConstraint (
                phase_col.begin(), phase_col.end()
            )
        ); // Add the mass balance constraint to ConstraintManager (mandatory)
    if ( activephases == 1 ) {
        // If only one phase is present, fix its corresponding variable index
        std::stringstream ss;
        ss << ( phase_col.begin() )->first << "_FRAC";
        fixed_indices.push_back ( main_indices.left.at ( ss.str() ) );
    }
    
    // TODO: Stop using phase_col below here and use only CompositionSet and sublattice_set

    // Add the sublattice site fraction constraints (mandatory)
    for ( auto i = phase_col.begin(); i != phase_col.end(); ++i ) {
        for ( auto j = i->second.get_sublattice_iterator(); j != i->second.get_sublattice_iterator_end(); ++j ) {
            std::vector<std::string> subl_list;
            for ( auto k = ( *j ).get_species_iterator(); k != ( *j ).get_species_iterator_end(); ++k ) {
                // Check if this species in this sublattice is on our list of elements to investigate
                if ( std::find ( conditions.elements.cbegin(),conditions.elements.cend(),*k ) != conditions.elements.cend() ) {
                    subl_list.push_back ( *k ); // Add to the list
                }
            }
            if ( subl_list.size() == 1 ) {
                std::stringstream ss;
                ss << i->first << "_" << std::distance ( i->second.get_sublattice_iterator(),j ) << "_" << * ( subl_list.begin() );
                fixed_indices.push_back ( main_indices.left.at ( ss.str() ) );
            }
            if ( subl_list.size() > 1 ) {
                cm.addConstraint (
                    SublatticeBalanceConstraint (
                        i->first,
                        std::distance ( i->second.get_sublattice_iterator(),j ),
                        subl_list.cbegin(),
                        subl_list.cend()
                    )
                );
            }
        }
    }

    // Add any user-specified constraints to the ConstraintManager

    for ( auto i = conditions.xfrac.cbegin(); i != conditions.xfrac.cend(); ++i ) {
        cm.addConstraint ( MassBalanceConstraint ( phase_col.begin(), phase_col.end(), i->first, i->second ) );
    }

    for ( auto i = cm.constraints.begin() ; i != cm.constraints.end(); ++i ) {
        BOOST_LOG_SEV ( opto_log, debug ) << "Constraint " << i->name << " LHS: " << i->lhs;
        BOOST_LOG_SEV ( opto_log, debug ) << "Constraint " << i->name << " RHS: " << i->rhs;
    }

    // Calculate first derivative ASTs of all constraints
    for ( auto i = main_indices.left.begin(); i != main_indices.left.end(); ++i ) {
        // for each variable, calculate derivatives of all the constraints
        for ( auto j = cm.constraints.begin(); j != cm.constraints.end(); ++j ) {
            boost::spirit::utree lhs = differentiate_utree ( j->lhs, i->first );
            boost::spirit::utree rhs = differentiate_utree ( j->rhs, i->first );
            lhs = simplify_utree ( lhs );
            rhs = simplify_utree ( rhs );
            if (
                ( lhs.which() == boost::spirit::utree_type::double_type || lhs.which() == boost::spirit::utree_type::int_type )
                &&
                ( rhs.which() == boost::spirit::utree_type::double_type || rhs.which() == boost::spirit::utree_type::int_type )
            ) {
                double lhsget, rhsget;
                lhsget = lhs.get<double>();
                rhsget = rhs.get<double>();
                if ( lhsget == rhsget ) {
                    continue;    // don't add zeros to the Jacobian
                }
            }
            boost::spirit::utree subtract_tree;
            subtract_tree.push_back ( "-" );
            subtract_tree.push_back ( lhs );
            subtract_tree.push_back ( rhs );
            int var_index = i->second;
            int cons_index = std::distance ( cm.constraints.begin(),j );
            jac_g_trees.push_back ( jacobian_entry ( cons_index,var_index,false,subtract_tree ) );
            BOOST_LOG_SEV ( opto_log, debug ) << "Jacobian of constraint  " << cons_index << " wrt variable " << var_index << " pre-calculated";
        }
    }

    // Add nonzero elements from objective Hessian to sparsity structure
    for ( auto i = comp_sets.cbegin(); i != comp_sets.cend(); ++i ) {
        std::set<std::list<int>> comp_set_hess_sparsity_structure = i->second.hessian_sparsity_structure ( main_indices );
        for ( auto j = comp_set_hess_sparsity_structure.cbegin(); j != comp_set_hess_sparsity_structure.cend(); ++j ) {
            hess_sparsity_structure.insert ( *j );
        }
    }

    // Calculate second derivatives of constraints
    for ( auto i = main_indices.left.begin(); i != main_indices.left.end(); ++i ) {
        // for each variable, calculate derivatives of the Jacobian w.r.t all the constraints
        for ( auto j = jac_g_trees.cbegin(); j != jac_g_trees.cend(); ++j ) {
            if ( i->second > j->var_index ) {
                continue;    // skip upper triangular
            }
            // second derivative of constraint jac_g_trees->cons_index w.r.t jac_g_trees->var_index, i->second
            boost::spirit::utree cons_second_deriv = differentiate_utree ( j->ast, i->first );
            //cons_second_deriv = simplify_utree(cons_second_deriv);
            hessian_set::iterator h_iter, h_end;
            // don't add zeros to the Hessian
            if ( is_zero_tree ( cons_second_deriv ) ) {
                continue;
            }

            h_iter = constraint_hessian_data.lower_bound ( boost::make_tuple ( i->second,j->var_index ) );
            h_end = constraint_hessian_data.upper_bound ( boost::make_tuple ( i->second,j->var_index ) );
            // create a new Hessian record if it does not exist
            if ( h_iter == h_end ) {
                h_iter = constraint_hessian_data.insert ( hessian_entry ( i->second,j->var_index ) ).first;
            }
            hessian_entry h_entry = *h_iter;
            h_entry.asts[j->cons_index] = cons_second_deriv; // set AST for constraint Hessian
            constraint_hessian_data.replace ( h_iter, h_entry ); // update original entry
            const std::list<int> nonzerolist {i->second, j->var_index};
            hess_sparsity_structure.insert ( nonzerolist ); // add nonzero to sparsity structure
            BOOST_LOG_SEV ( opto_log, debug ) << "Hessian of constraint  "
                                              << j->cons_index << " (" << i->second << "," << j->var_index << ") pre-calculated";
        }
    }

    BOOST_LOG_SEV ( opto_log, debug ) << "function exit";
}

GibbsOpt::~GibbsOpt()
{}
// kate: indent-mode cstyle; indent-width 4; replace-tabs on; 
