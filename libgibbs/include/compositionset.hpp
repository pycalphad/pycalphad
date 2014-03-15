/*=============================================================================
	Copyright (c) 2012-2014 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// declaration for CompositionSet class

#ifndef COMPOSITIONSET_INCLUDED
#define COMPOSITIONSET_INCLUDED

#include "libgibbs/include/models.hpp"
#include "libgibbs/include/constraint.hpp"
#include "libgibbs/include/optimizer/ast_set.hpp"
#include "libgibbs/include/conditions.hpp"
#include "libgibbs/include/utils/ast_caching.hpp"
#include "libtdb/include/structure.hpp"
#include <boost/bimap.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <memory>
#include <set>
#include <list>
#include <vector>
#include <utility>

// A CompositionSet works with libtdb's Phase class
// Its purpose is to handle the optimizer's specific configuration for the given conditions and models
// Multiple CompositionSets of the same Phase can be created to handle miscibility gaps
class CompositionSet
{
public:
    double evaluate_objective ( evalconditions const&, boost::bimap<std::string, int> const &, double* const ) const;
    double evaluate_objective ( evalconditions const &, std::map<std::string,double> const & ) const;
    std::map<int,double> evaluate_objective_gradient (
        evalconditions const&, boost::bimap<std::string, int> const &, double* const ) const;
    std::map<int,double> evaluate_objective_gradient (
        evalconditions const &, std::map<std::string,double> const & ) const;
    std::vector<double> evaluate_internal_objective_gradient (
        evalconditions const& conditions, double* const ) const;
    std::map<std::list<int>,double> evaluate_objective_hessian (
        evalconditions const&, boost::bimap<std::string, int> const &, double* const ) const;
    boost::numeric::ublas::symmetric_matrix<double,boost::numeric::ublas::lower> evaluate_objective_hessian_matrix (
        evalconditions const& conditions,
        boost::bimap<std::string, int> const &main_indices,
        std::vector<double> const &x ) const;
    std::set<std::list<int>> hessian_sparsity_structure ( boost::bimap<std::string, int> const & ) const;

    // make CompositionSet from existing Phase
    CompositionSet (
        const Phase &phaseobj,
        const parameter_set &pset,
        const sublattice_set &sublset,
        boost::bimap<std::string, int> const &main_indices );

    // make CompositionSet from another CompositionSet; used for miscibility gaps
    // this will create a copy
    CompositionSet (
        const CompositionSet &other,
        const std::map<std::string,double> &new_starting_point,
        const std::string &new_name ) {
        std::string old_phase_name, new_phase_name;
        // These are specified by the user
        cset_name = new_name;
        starting_point = new_starting_point;

        // Copy everything else from the parent CompositionSet
        // Deep copy the model map
        for ( auto energymod = other.models.begin(); energymod != other.models.end(); ++energymod ) {
            models.insert ( std::make_pair ( energymod->first,energymod->second->clone_with_renamed_phase ( old_phase_name, new_phase_name ) ) );
        }
        jac_g_trees = other.jac_g_trees;
        hessian_data = other.hessian_data;
        tree_data = other.tree_data;
        first_derivatives = other.first_derivatives;
        symbols = other.symbols;
        cm = other.cm;
        phase_indices = other.phase_indices;
        constraint_null_space_matrix = other.constraint_null_space_matrix;
    }

    CompositionSet() { }

    CompositionSet ( CompositionSet &&other ) {
        cset_name = std::move ( other.cset_name );
        models = std::move ( other.models );
        jac_g_trees = std::move ( other.jac_g_trees );
        hessian_data = std::move ( other.hessian_data );
        tree_data = std::move ( other.tree_data );
        first_derivatives = std::move ( other.first_derivatives );
        symbols = std::move ( other.symbols );
        cm = std::move ( other.cm );
        phase_indices = std::move ( other.phase_indices );
        constraint_null_space_matrix = std::move ( other.constraint_null_space_matrix );
        starting_point = std::move ( starting_point );
    }
    CompositionSet& operator= ( CompositionSet &&other ) {
        cset_name = std::move ( other.cset_name );
        models = std::move ( other.models );
        jac_g_trees = std::move ( other.jac_g_trees );
        hessian_data = std::move ( other.hessian_data );
        tree_data = std::move ( other.tree_data );
        first_derivatives = std::move ( other.first_derivatives );
        symbols = std::move ( other.symbols );
        cm = std::move ( other.cm );
        phase_indices = std::move ( other.phase_indices );
        constraint_null_space_matrix = std::move ( other.constraint_null_space_matrix );
        starting_point = std::move ( starting_point );
    }
    const std::vector<jacobian_entry>& get_jacobian() const {
        return jac_g_trees;
    };
    const std::vector<Constraint>& get_constraints() const {
        return cm.constraints;
    };
    const boost::bimap<std::string, int>& get_variable_map() const {
        return phase_indices;
    };
    const boost::numeric::ublas::matrix<double>& get_constraint_null_space_matrix() const {
        return constraint_null_space_matrix;
    };
    const ASTSymbolMap& get_symbols() const {
        return symbols;
    };

    void set_name ( const std::string &name ) {
        cset_name = name;
    }
    std::string name() const {
        return cset_name;
    }
    void set_starting_point ( std::map<std::string,double> &startx ) {
        starting_point = startx;
    }
    std::map<std::string,double> get_starting_point() const {
        return starting_point;
    }

    CompositionSet ( const CompositionSet & ) = delete;
    CompositionSet& operator= ( const CompositionSet & ) = delete;
private:
    std::string cset_name;
    std::map<std::string,double> starting_point; // starting point for optimizing this composition set
    std::map<std::string,std::unique_ptr<EnergyModel>> models;
    std::map<int,boost::spirit::utree> first_derivatives;
    boost::bimap<std::string, int> phase_indices;
    std::vector<jacobian_entry> jac_g_trees;
    hessian_set hessian_data;
    ast_set tree_data;
    ASTSymbolMap symbols; // maps special symbols to ASTs and their derivatives
    ConstraintManager cm; // handles constraints internal to the phase, e.g., site fraction balances
    void build_constraint_basis_matrices ( sublattice_set const &sublset );
    boost::numeric::ublas::matrix<double> constraint_null_space_matrix;
};

#endif
// kate: indent-mode cstyle; indent-width 4; replace-tabs on; 
