/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

	Based on example code from Boost MultiIndex.
	Copyright (c) 2003-2008 Joaquin M Lopez Munoz.
	See http://www.boost.org/libs/multi_index for library home page.

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition for ast_set objects

#ifndef INCLUDED_AST_SET
#define INCLUDED_AST_SET

#include <string>
#include <sstream>
#include <boost/spirit/include/support_utree.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/mem_fun.hpp>

struct ast_entry {
	std::list<std::string> diffvars; // variables of differentiation
	std::string model_name;
	boost::spirit::utree ast;
	ast_entry (
			std::list<std::string> dv, std::string mod_name, boost::spirit::utree tree) :
				diffvars(dv),
				model_name(mod_name),
				ast(tree) {}
	std::list<std::string>::size_type ast_derivative_order() const {
		return diffvars.size();
	}
};

struct ast_deriv_order_index {};
struct ast_model_name_index {};
struct ast_diffvars_index {};

typedef boost::multi_index::multi_index_container<
		ast_entry,
		boost::multi_index::indexed_by<
	    boost::multi_index::ordered_non_unique<
	    boost::multi_index::tag<ast_deriv_order_index>,
	      BOOST_MULTI_INDEX_CONST_MEM_FUN(ast_entry,std::list<std::string>::size_type,ast_derivative_order)
	    >,
		boost::multi_index::ordered_non_unique<
		boost::multi_index::tag<ast_model_name_index>,
		BOOST_MULTI_INDEX_MEMBER(ast_entry,std::string,model_name)
		>,
		boost::multi_index::ordered_non_unique<
		boost::multi_index::tag<ast_diffvars_index>,
		BOOST_MULTI_INDEX_MEMBER(ast_entry,std::list<std::string>,diffvars)
		>
>
> ast_set;


typedef boost::multi_index::multi_index_container<
		const ast_entry*,
		boost::multi_index::indexed_by<
	    boost::multi_index::ordered_non_unique<
	    boost::multi_index::tag<ast_deriv_order_index>,
	      BOOST_MULTI_INDEX_CONST_MEM_FUN(ast_entry,std::list<std::string>::size_type,ast_derivative_order)
	    >,
		boost::multi_index::ordered_non_unique<
		boost::multi_index::tag<ast_model_name_index>,
		BOOST_MULTI_INDEX_MEMBER(ast_entry,std::string,model_name)
		>,
		boost::multi_index::ordered_non_unique<
		boost::multi_index::tag<ast_diffvars_index>,
		BOOST_MULTI_INDEX_MEMBER(ast_entry,std::list<std::string>,diffvars)
		>
>
> ast_set_view;

#endif
