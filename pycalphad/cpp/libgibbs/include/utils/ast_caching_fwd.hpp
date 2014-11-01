/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// forward declaration for objects related to AST caching

#ifndef INCLUDED_AST_CACHING_FWD
#define INCLUDED_AST_CACHING_FWD

#include <map>
#include <string>

struct CachedAbstractSyntaxTree;
typedef std::map<std::string, CachedAbstractSyntaxTree> ASTSymbolMap;

#endif
