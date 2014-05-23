/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis

 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Deep copy/rename (DCR) implementation for EnergyModel

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/utils/ast_container_rename.hpp"
#include "libgibbs/include/utils/ast_variable_rename.hpp"
#include "libgibbs/include/models.hpp"
#include "libtdb/include/logging.hpp"
#include <memory>

std::unique_ptr<EnergyModel> EnergyModel::clone_with_renamed_phase (
    const std::string &old_phase_name,
    const std::string &new_phase_name
) const
{
    BOOST_LOG_NAMED_SCOPE ( "EnergyModel::clone_with_renamed_phase" );
    logger opto_log ( journal::keywords::channel = "optimizer" );
    BOOST_LOG_SEV( opto_log, debug ) << "entered";
    auto copymodel = std::unique_ptr<EnergyModel> ( new EnergyModel ( *this ) );
    BOOST_LOG_SEV( opto_log, debug ) << "created EnergyModel copy tethered to unique_ptr";
    // Deep copy and rename the ast_symbol_table
    ASTSymbolMap new_map ( ast_copy_with_renamed_phase ( ast_symbol_table, old_phase_name, new_phase_name ) );
    BOOST_LOG_SEV( opto_log, debug ) << "DCR ASTSymbolMap";
    // Perform a deep rename on model_ast
    ast_variable_rename ( copymodel->model_ast, old_phase_name, new_phase_name );
    BOOST_LOG_SEV( opto_log, debug ) << "DCR model_ast(" 
        << old_phase_name << " -> " << new_phase_name << ") = " << copymodel->model_ast;
    copymodel->ast_symbol_table = std::move ( new_map );
    BOOST_LOG_SEV( opto_log, debug ) << "returning";
    return std::move ( copymodel );
}
