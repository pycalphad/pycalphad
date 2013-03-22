/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// fixture_func.hpp -- header file for TDB function parser text fixture

#ifndef FIXTURE_FUNC_HPP_
#define FIXTURE_FUNC_HPP_

#include "libtdb/include/grammars/function_grammar.hpp"
#include "test/include/fixtures/fixture_math.hpp"

struct FuncParserFixture : public MathParserFixture
{
	FuncParserFixture()
		: func_parser(macros,statevars)
	{
		// Handled in the inherited constructor
	}

	double func_eval(const std::string &funcexpr)
	{
		using boost::spirit::ascii::space;
		using boost::spirit::utree;
		typedef boost::spirit::utree_type utree_type;
		typedef std::string::const_iterator iterator_type;

		utree ret_tree; // return storage for abstract syntax tree

		// Initialize the iterators for the string
		iterator_type iter = funcexpr.begin();
		iterator_type end = funcexpr.end();

		// Parse the string and put the abstract syntax tree in ret_tree
		bool r = phrase_parse(iter, end, func_parser, space, ret_tree);

		if (r && iter == end)
		{
			// Get the processed abstract syntax tree and determine the value
			utree final_tree = process_utree(ret_tree, conditions);
			if (final_tree.which() == utree_type::double_type) {
				return final_tree.get<double>();
			}
			else {
				BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo("Bad abstract syntax tree"));
			}
		}
		else
		{
			std::string::const_iterator some = iter+30;
			std::string context(iter, (some>end)?end:some);
			std::string errmsg("Syntax error: " + context + "...");
			BOOST_THROW_EXCEPTION(syntax_error() << specific_errinfo(errmsg));
		}
		return 0; // impossible
	}
	double add_macro(const std::string &macroname, const std::string &funcexpr)
	{
		using boost::spirit::ascii::space;
		using boost::spirit::utree;
		typedef boost::spirit::utree_type utree_type;
		typedef std::string::const_iterator iterator_type;

		utree ret_tree; // return storage for abstract syntax tree

		// Initialize the iterators for the string
		iterator_type iter = funcexpr.begin();
		iterator_type end = funcexpr.end();

		// Parse the string and put the abstract syntax tree in ret_tree
		bool r = phrase_parse(iter, end, func_parser, space, ret_tree);

		if (r && iter == end)
		{
			macros.add(macroname, ret_tree);
		}
		else
		{
			std::string::const_iterator some = iter+30;
			std::string context(iter, (some>end)?end:some);
			std::string errmsg("Syntax error: " + context + "...");
			BOOST_THROW_EXCEPTION(syntax_error() << specific_errinfo(errmsg));
		}
		return 0; // impossible
	}
	function_grammar func_parser;
};


#endif /* FIXTURE_FUNC_HPP_ */
