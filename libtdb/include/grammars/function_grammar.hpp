/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// function_grammar.hpp -- header file for TDB file T-dependent FUNCTIONs and associated grammar

#ifndef INCLUDED_FUNCTION_GRAMMAR
#define INCLUDED_FUNCTION_GRAMMAR

#include "libtdb/include/warning_disable.hpp"
#include <string>
#include <boost/spirit/include/support_utree.hpp>
#include <boost/spirit/include/qi.hpp>
#include "libtdb/include/grammars/math_grammar.hpp"

struct function_grammar :
	boost::spirit::qi::grammar<std::string::const_iterator, boost::spirit::utree(), boost::spirit::ascii::space_type>
{

	function_grammar(const boost::spirit::qi::symbols<char, boost::spirit::utree>& functions, 
		const boost::spirit::qi::symbols<char, boost::spirit::utree>& variables);
	//boost::spirit::qi::rule<std::string::const_iterator, boost::spirit::qi::locals<std::string>, std::string()> text;
	boost::spirit::qi::rule<std::string::const_iterator, void(), boost::spirit::ascii::space_type> endstring;
	boost::spirit::qi::rule<std::string::const_iterator, void(), boost::spirit::ascii::space_type> randomtext;
	boost::spirit::qi::rule<std::string::const_iterator, boost::spirit::utree(), boost::spirit::ascii::space_type>
		mathexpr, expressiontree, start;
	boost::spirit::qi::rule
		<std::string::const_iterator, std::vector<boost::variant<double, boost::spirit::utree>>(), boost::spirit::ascii::space_type> 
		expressions, firstexpression, expression, firstexpnomax, lastexpnomax;
	calculator mycalc; // defined in math_grammar.cpp
};

#endif
