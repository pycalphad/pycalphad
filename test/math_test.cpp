// math_test.cpp -- test suite for the arithmetic parser

#include "test_pch.h"
#include "tdbread/warning_disable.h"
#include "tdbread/exceptions.h"
#include "tdbread/math_grammar.h"
#include "tdbread/math_expr.h"
#include "tdbread/conditions.h"
#include <string>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/spirit/include/support_utree.hpp>
#include <boost/spirit/include/qi.hpp>

struct MathParserFixture
{
	MathParserFixture()
		: calc_parser(macros,statevars)
	{
	}
	double calculate(const std::string &mathexpr)
	{
		using boost::spirit::ascii::space;
		using boost::spirit::utree;
		typedef std::string::const_iterator iterator_type;

		utree ret_tree; // abstract syntax tree for math expressions

		// Initialize the iterators for the string
		iterator_type iter = mathexpr.begin();
		iterator_type end = mathexpr.end();

		// Parse the string and put the abstract syntax tree in ret_tree
		bool r = phrase_parse(iter, end, calc_parser, space, ret_tree);

		if (r && iter == end)
		{
			try {
				// Walk the abstract syntax tree and determine the value
				double ret_val = process_utree(ret_tree, conditions).get<double>();
				return ret_val;
			}
			catch (std::exception &e) {
				throw e; // push the exception up the callstack for the test framework
			}
		}
		else
		{
			std::string::const_iterator some = iter+30;
			std::string context(iter, (some>end)?end:some);
			std::string errmsg("Syntax error: " + context + "...");
			BOOST_THROW_EXCEPTION(syntax_error() << specific_errinfo(errmsg));
		}
	}
	calculator calc_parser;
	boost::spirit::qi::symbols<char, boost::spirit::utree> macros; // all of the macros (FUNCTIONs in Thermo-Calc lingo)
	boost::spirit::qi::symbols<char, boost::spirit::utree> statevars; // all valid state variables
	evalconditions conditions;
};

BOOST_FIXTURE_TEST_SUITE(MathParserSuite, MathParserFixture)

BOOST_AUTO_TEST_CASE(SimpleAddition)
{
	BOOST_REQUIRE_EQUAL(calculate("2+6"), 8);
}
BOOST_AUTO_TEST_CASE(SimpleAdditionOfNegatives)
{
	BOOST_REQUIRE_EQUAL(calculate("-2+6"), 4);
}

BOOST_AUTO_TEST_SUITE_END()