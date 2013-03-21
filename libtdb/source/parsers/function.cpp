/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// function.cpp -- parser for FUNCTION command

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/database_tdb.hpp"
#include "libtdb/include/grammars/function_grammar.hpp"
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_function.hpp>
#include <boost/spirit/include/support_utree.hpp>

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace spirit = boost::spirit;


void Database::DatabaseTDB::Function(std::string &argstr) {
    using boost::spirit::ascii::space;
    using boost::spirit::utree;
    typedef std::string::const_iterator iterator_type;
	utree ret_tree;

	function_grammar func_grammar (macros, statevars); 
	auto namefind = boost::find_first(argstr, " ");
	std::string func_name(argstr.begin(), namefind.begin());

	iterator_type iter = namefind.end(); // start iterator after function name
	iterator_type end = argstr.end();
	//std::cout << "parsing" << std::endl;
	bool r = phrase_parse(iter, end, func_grammar, space, ret_tree);
	//std::cout << "parse out" << std::endl;
	if (r && iter == end)
	{
		macros.add(func_name, ret_tree); // add the abstract syntax tree for "func_name" to the symbol table of the DB
		//std::cout << "ADDED SYMBOL: " << func_name << std::endl;
	}
	else
	{
		std::string::const_iterator some = iter+30;
		std::string context(iter, (some>end)?end:some);
		std::string errmsg("Syntax error: " + context + "...");
		BOOST_THROW_EXCEPTION(syntax_error() << specific_errinfo(errmsg));
	}
}
