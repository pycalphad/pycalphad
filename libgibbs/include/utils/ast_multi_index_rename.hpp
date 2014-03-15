/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis

 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Abstract syntax tree boost::multi_index renaming (for duplicate phases generated from miscibility gaps)

#ifndef INCLUDED_AST_MULTI_INDEX_RENAME
#define INCLUDED_AST_MULTI_INDEX_RENAME

#include "libgibbs/include/constraint.hpp"
#include "libgibbs/include/optimizer/ast_set.hpp"
#include "libgibbs/include/utils/ast_variable_rename.hpp"
#include "libgibbs/include/utils/ast_container_rename.hpp"

// Rename a hessian_set of abstract syntax trees with a new phase name
inline hessian_set ast_copy_with_renamed_phase (
    const hessian_set &old_set,
    const std::string &old_phase_name,
    const std::string &new_phase_name
)
{
    hessian_set new_set;
    for ( const auto &old_record : old_set ) {
        auto new_entry ( old_record );
        new_entry.asts = ast_copy_with_renamed_phase ( old_record.asts,old_phase_name, new_phase_name );
        new_set.insert ( std::move ( new_entry ) );
    }
    return new_set;
}

// Rename an ast_set of abstract syntax trees with a new phase name
inline ast_set ast_copy_with_renamed_phase (
    const ast_set &old_set,
    const std::string &old_phase_name,
    const std::string &new_phase_name
)
{
    ast_set new_set;
    for ( const auto &old_record : old_set ) {
        auto new_entry ( old_record );
        ast_variable_rename ( new_entry.ast,old_phase_name, new_phase_name );
        new_entry.diffvars = ast_copy_with_renamed_phase ( old_record.diffvars,old_phase_name,new_phase_name );
        new_set.insert ( std::move ( new_entry ) );
    }
    return new_set;
}

// Rename a vector of jacobian_entry abstract syntax trees with a new phase name
inline std::vector<jacobian_entry> ast_copy_with_renamed_phase (
    const std::vector<jacobian_entry> &old_set,
    const std::string &old_phase_name,
    const std::string &new_phase_name
)
{
    std::vector<jacobian_entry> new_set;
    for ( const auto &old_record : old_set ) {
        auto new_entry ( old_record );
        ast_variable_rename ( new_entry.ast,old_phase_name, new_phase_name );
        new_set.push_back(new_entry);
    }
    return new_set;
}

#endif
