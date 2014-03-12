/*=============================================================================
 Copyright* (c) 2012-2014 Richard Otis

 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/utils/ast_variable_rename.hpp"
#include "libtdb/include/exceptions.hpp"
#include "libtdb/include/logging.hpp"
#include <boost/spirit/include/support_utree.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <string>

// Modify abstract syntax tree variable names to use a new prefix (e.g., FCC_A1_0_VA -> FCC_A1#2_0_VA)
void ast_variable_rename (
    boost::spirit::utree & ut,
    std::string const & old_prefix,
    std::string const & new_prefix )
{
    BOOST_LOG_NAMED_SCOPE ( "ast_variable_rename" );
    typedef boost::spirit::utree utree;
    typedef boost::spirit::utree_type utree_type;
    switch ( ut.which() ) {
    case utree_type::list_type: {
        auto it = ut.begin();
        auto end = ut.end();
        while ( it != end ) {
            if ( ( *it ).which() == utree_type::string_type ) {
                // operator/function
                boost::spirit::utf8_string_range_type rt = ( *it ).get<boost::spirit::utf8_string_range_type>();
                std::string op ( rt.begin(), rt.end() ); // set the symbol

                // Check if the string is a variable name that starts with old_prefix
                // if so, rename it
                if ( boost::algorithm::istarts_with ( op, old_prefix ) ) {
                    boost::algorithm::ireplace_first ( op, old_prefix, new_prefix );
                    utree new_var ( op );
                    ( *it ).swap ( new_var ); // Replace with the new variable
                }

                // step through the range check operator
                if ( op == "@" ) {
                    ++it;
                    try {
                        // curT
                        ast_variable_rename ( *it, old_prefix, new_prefix );
                        ++it;
                        // lowlimit
                        ast_variable_rename ( *it, old_prefix, new_prefix );
                        ++it;
                        // highlimit
                        ast_variable_rename ( *it, old_prefix, new_prefix );
                    } catch ( boost::exception &e ) {
                        e << ast_errinfo ( *it );
                        throw;
                    }
                }

                ++it; // get left-hand side
                auto lhsiter = it;
                ++it; // get right-hand side
                auto rhsiter = it;
                if ( lhsiter != end ) {
                    ast_variable_rename ( *lhsiter, old_prefix, new_prefix );
                }
                if ( rhsiter != end ) {
                    ast_variable_rename ( *rhsiter, old_prefix, new_prefix );
                }
            }

            ++it; // next entity (if any)
        }
    }
    case utree_type::string_type: {
        // Check if the string is a variable name that starts with old_prefix
        // if so, rename it
        boost::spirit::utf8_string_range_type rt = ut.get<boost::spirit::utf8_string_range_type>();
        std::string varname ( rt.begin(), rt.end() );
        if ( boost::algorithm::istarts_with ( varname, old_prefix ) ) {
            boost::algorithm::ireplace_first ( varname, old_prefix, new_prefix );
            utree new_var ( varname );
            ut.swap ( new_var ); // Replace with the new variable
        }
    }
    }
    return;
}
