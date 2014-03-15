/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis

 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Abstract syntax tree map renaming (for duplicate phases generated from miscibility gaps)

#ifndef INCLUDED_AST_MAP_RENAME
#define INCLUDED_AST_MAP_RENAME
#include "libgibbs/include/utils/ast_caching.hpp"
#include "libgibbs/include/utils/ast_variable_rename.hpp"
#include <boost/algorithm/string/replace.hpp>
#include <map>
#include <string>

// Rename a map of abstract syntax trees with a new phase name
template<typename IndexType,typename EntryType>
std::map<IndexType, EntryType> ast_copy_with_renamed_phase (
    const std::map<IndexType,EntryType> &old_map,
    const std::string &old_phase_name,
    const std::string &new_phase_name
)
{
    std::map<IndexType, EntryType> new_map;
    for ( const auto &old_record : old_map ) {
        EntryType new_entry ( old_record.second );
        ast_variable_rename ( new_entry,old_phase_name, new_phase_name );
        new_map.emplace ( std::make_pair ( old_record.first, std::move ( new_entry ) ) );
    }
    return new_map;
}

// Rename a map of abstract syntax trees with a new phase name
// Specialization for string-based AST maps
template<typename EntryType>
std::map<std::string, EntryType> ast_copy_with_renamed_phase (
    const std::map<std::string,EntryType> &old_map,
    const std::string &old_phase_name,
    const std::string &new_phase_name
)
{
    std::map<std::string, EntryType> new_map;
    for ( const auto &old_record : old_map ) {
        std::string entry_name ( old_record.first );
        EntryType new_entry ( old_record.second );
        boost::algorithm::ireplace_first ( entry_name, old_phase_name, new_phase_name );
        ast_variable_rename ( new_entry,old_phase_name, new_phase_name );
        new_map.emplace ( std::make_pair ( entry_name, std::move ( new_entry ) ) );
    }
    return new_map;
}

// Rename a map of abstract syntax trees with a new phase name
// Specialization for string-based cached AST maps
template<>
inline std::map<std::string, CachedAbstractSyntaxTree> ast_copy_with_renamed_phase (
    const std::map<std::string,CachedAbstractSyntaxTree> &old_map,
    const std::string &old_phase_name,
    const std::string &new_phase_name
)
{
    std::map<std::string, CachedAbstractSyntaxTree> new_map;
    for ( const auto &old_record : old_map ) {
        std::string entry_name ( old_record.first );
        boost::spirit::utree new_entry ( old_record.second.get() );
        boost::algorithm::ireplace_first ( entry_name, old_phase_name, new_phase_name );
        ast_variable_rename ( new_entry,old_phase_name, new_phase_name );
        CachedAbstractSyntaxTree new_cached_ast ( new_entry );
        new_map.emplace ( std::make_pair ( entry_name, std::move ( new_cached_ast ) ) );
    }
    return new_map;
}
#endif
