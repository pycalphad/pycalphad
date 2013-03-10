// math_grammar.cpp -- grammar construction for arithmetic expressions
// Based on http://www.boost.org/doc/libs/1_51_0/libs/spirit/example/qi/calc_utree_ast.cpp

/*=============================================================================
    Copyright (c) 2001-2011 Hartmut Kaiser
    Copyright (c) 2001-2011 Joel de Guzman
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

//#define BOOST_SPIRIT_DEBUG
#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/grammars/math_grammar.hpp"
#include <boost/spirit/include/support_utree.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_function.hpp>

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace spirit = boost::spirit;

struct expr
{
	template <typename T1, typename T2 = void>
	struct result { typedef void type; };

	expr(char* op) : op(op) {}

	void operator()(spirit::utree& expr, spirit::utree const& rhs) const
	{
		spirit::utree lhs;
		lhs.swap(expr);
		expr.push_back(spirit::utf8_string_range_type(op,op + strlen(op)));
		expr.push_back(lhs);
		expr.push_back(rhs);
	}

	char const* op;
};
boost::phoenix::function<expr> const plus = expr("+");
boost::phoenix::function<expr> const minus = expr("-");
boost::phoenix::function<expr> const power = expr("**");
boost::phoenix::function<expr> const times = expr("*");
boost::phoenix::function<expr> const divide = expr("/");

struct negate_expr
{
	template <typename T1, typename T2 = void>
	struct result { typedef void type; };

	void operator()(spirit::utree& expr, spirit::utree const& rhs) const
	{
		char const* op = "-";
		expr.clear();
		expr.push_back(boost::spirit::utf8_string_range_type(op, op + strlen(op)));
		expr.push_back(rhs);
	}
};
boost::phoenix::function<negate_expr> neg;

struct natlog_expr
{
	template <typename T1, typename T2 = void>
	struct result { typedef void type; };

	void operator()(spirit::utree& expr, spirit::utree const& rhs) const
	{
		char const* op = "ln";
		expr.clear();
		expr.push_back(boost::spirit::utf8_string_range_type(op, op + strlen(op)));
		expr.push_back(rhs);
	}
};
boost::phoenix::function<natlog_expr> natlog;

struct exp_expr
{
	template <typename T1, typename T2 = void>
	struct result { typedef void type; };

	void operator()(spirit::utree& expr, spirit::utree const& rhs) const
	{
		char const* op = "exp";
		expr.clear();
		expr.push_back(boost::spirit::utf8_string_range_type(op, op + strlen(op)));
		expr.push_back(rhs);
	}
};
boost::phoenix::function<exp_expr> expn;

///////////////////////////////////////////////////////////////////////////////
//  Our calculator grammar
///////////////////////////////////////////////////////////////////////////////
calculator::calculator(const qi::symbols<char, spirit::utree>& functions, const qi::symbols<char, spirit::utree>& variables) : 
	qi::grammar<std::string::const_iterator, ascii::space_type, spirit::utree()>::base_type(expression) {
			using qi::double_;
			using qi::char_;
			using ascii::alpha;
			using ascii::alnum;
			using qi::_val;
			using qi::_1;
			using qi::lit;
			using qi::lexeme;
			using qi::eps;
			using ascii::string;
			using ascii::no_case;

			expression =
				term                            [_val = _1]
			>> *(   ('+' >> term            [plus(_val, _1)])
				|   ('-' >> term            [minus(_val, _1)])
				)
				;

			term =
				factor                          [_val = _1]
			>> *(   ('*' >> !lit('*') >> factor          [times(_val, _1)])
				|   ('/' >> factor          [divide(_val, _1)])
				)
				;

			factor =
				(
				double_                           [_val = _1]
				|   (functions [_val = _1]  >> -lit('#'))
				|   variables                     [_val = _1]
				|   '(' >> expression           [_val = _1] >> ')'
				|   ('-' >> factor              [neg(_val, _1)])
				|   ('+' >> factor              [_val = _1])
				|   (no_case["ln"] >> factor           [natlog(_val, _1)])
				|   (no_case["log"] >> factor          [natlog(_val, _1)]) // Thermo-Calc says log() is the same as ln()
				|   (no_case["exp"] >> factor          [expn(_val, _1)])
				)
				>>   *("**" >> factor                    [power(_val, _1)])
				;

			BOOST_SPIRIT_DEBUG_NODE(expression);
			BOOST_SPIRIT_DEBUG_NODE(term);
			BOOST_SPIRIT_DEBUG_NODE(factor);
}