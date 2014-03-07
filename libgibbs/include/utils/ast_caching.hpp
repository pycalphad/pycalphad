/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// object that can be inserted into utree-based abstract syntax trees (ASTs)

#ifndef INCLUDED_AST_CACHING
#define INCLUDED_AST_CACHING

#include "libgibbs/include/utils/ast_caching_fwd.hpp"
#include "libgibbs/include/utils/math_expr.hpp"
#include <boost/spirit/include/support_utree.hpp>
#include <map>
#include <string>

/*
 * From magnetic model problems:
 * Idea for dealing with slowness caused by repeated tau differentiation:
 * Make "TAU" (needs phase-dependent name) map to a free function that returns the appropriate utree
 * Differentiation of "TAU" triggers another function, a wrapper for differentiate_utree()
 * EXCEPT this time we save the result tree to a cache map for each variable
 * Future calls to differentiate "TAU" will hit the cache instead of the expensive call
 * This should be useful for trees that tend to repeat in models
 */

struct CachedAbstractSyntaxTree {
	CachedAbstractSyntaxTree(boost::spirit::utree const &ut) : ast(ut) { }
	boost::spirit::utree const& get() const { return ast; };
	boost::spirit::utree const& differentiate(std::string const &variable, ASTSymbolMap const &symbols) const {
		const auto cache_find = differentiated_ast_cache.find(variable);
		const auto cache_end = differentiated_ast_cache.end();
		if (cache_find != cache_end) {
			// cache hit: return the previously calculated AST
			return cache_find->second;
		}
		else {
			// cache miss: perform the differentiation and store the result
			auto result = differentiated_ast_cache.emplace(variable, differentiate_utree(ast, variable, symbols));
			return result.first->second; // return const
		}
	}
private:
	const boost::spirit::utree ast; // the cached AST
	mutable std::map<std::string,const boost::spirit::utree> differentiated_ast_cache; // cached AST w.r.t variables
};

#endif
