/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis

 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Declaration for AST variable renaming

#ifndef INCLUDED_AST_VARIABLE_RENAME
#define INCLUDED_AST_VARIABLE_RENAME

#include "libtdb/include/warning_disable.hpp"
#include <boost/spirit/home/support/utree/utree_traits_fwd.hpp>
#include <string>

// Modify abstract syntax tree variable names to use a new prefix (e.g., FCC_A1_0_VA -> FCC_A1#2_0_VA)
void ast_variable_rename (
    boost::spirit::utree & ut,
    std::string const & old_prefix,
    std::string const & new_prefix );
#endif
