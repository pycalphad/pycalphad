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
		// Without these calls, there are "access violation"
		// runtime errors whenever calc_parser tries to match
		// a rule that involves these qi::symbols objects.
		// I don't *think* this case (due to initialization order)
		// can happen in production, so we'll leave it for now.
		macros.clear();
		statevars.clear();

		conditions.statevars.clear();
	}
	void add_state_variable(const std::string &s) {
		statevars.add(s,boost::spirit::utree(s.c_str()));
	}
	void clear_conditions() {
		macros.clear();
		statevars.clear();
		conditions.statevars.clear();
	}
	void set_conditions(const std::string &var, const double val) {
		add_state_variable(var);
		conditions.statevars[var.c_str()[0]] = val;
	}
	double calculate(const std::string &mathexpr)
	{
		using boost::spirit::ascii::space;
		using boost::spirit::utree;
		typedef boost::spirit::utree_type utree_type;
		typedef std::string::const_iterator iterator_type;

		utree ret_tree; // abstract syntax tree for math expressions

		// Initialize the iterators for the string
		iterator_type iter = mathexpr.begin();
		iterator_type end = mathexpr.end();

		// Parse the string and put the abstract syntax tree in ret_tree
		bool r = phrase_parse(iter, end, calc_parser, space, ret_tree);

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
	}
	boost::spirit::qi::symbols<char, boost::spirit::utree> macros; // all of the macros (FUNCTIONs in Thermo-Calc lingo)
	boost::spirit::qi::symbols<char, boost::spirit::utree> statevars; // all valid state variables
	calculator calc_parser;
	evalconditions conditions;
};

BOOST_FIXTURE_TEST_SUITE(MathParserSuite, MathParserFixture)

	BOOST_AUTO_TEST_SUITE(MathParserSimpleOperations)
		BOOST_AUTO_TEST_CASE(Addition)
		{
			BOOST_REQUIRE_EQUAL(calculate("2+6"), 8);
		}
		BOOST_AUTO_TEST_CASE(AdditionWithNegation)
		{
			BOOST_REQUIRE_EQUAL(calculate("-21+6"), -15);
		}
		BOOST_AUTO_TEST_CASE(Subtraction)
		{
			BOOST_REQUIRE_EQUAL(calculate("4-30"), -26);
		}
		BOOST_AUTO_TEST_CASE(Multiplication)
		{
			BOOST_REQUIRE_EQUAL(calculate("7*3"), 21);
		}
		BOOST_AUTO_TEST_CASE(Division)
		{
			BOOST_REQUIRE_EQUAL(calculate("400/2"), 200);
		}
		BOOST_AUTO_TEST_CASE(ExponentialFunction)
		{
			BOOST_REQUIRE_CLOSE_FRACTION(calculate("EXP(1)"), 2.7182818284590451, 1e-15);
			BOOST_REQUIRE_CLOSE_FRACTION(calculate("eXp(1)"), 2.7182818284590451, 1e-15);
		}
		BOOST_AUTO_TEST_CASE(NaturalLogarithm)
		{
			BOOST_REQUIRE_CLOSE_FRACTION(calculate("LN(2.7182818284590451)"), 1, 1e-15);
			BOOST_REQUIRE_CLOSE_FRACTION(calculate("lN(2.7182818284590451)"), 1, 1e-15);
		}
		BOOST_AUTO_TEST_CASE(Exponentiation)
		{
			BOOST_REQUIRE_EQUAL(calculate("13**7"), 62748517);
		}
		BOOST_AUTO_TEST_CASE(ExponentiationRespectsOrderOfOperations)
		{
			BOOST_REQUIRE_EQUAL(calculate("5*3**7"), 10935);
			BOOST_REQUIRE_EQUAL(calculate("(5*3)**7"), 170859375);
			BOOST_REQUIRE_EQUAL(calculate("(5*3)**7/2"), 85429687.5);
			BOOST_REQUIRE_CLOSE_FRACTION(calculate("(5*3)**(7/2)"), 13071.318793450031, 1e-15);
		}
		BOOST_AUTO_TEST_CASE(StateVariables)
		{
			clear_conditions();
			set_conditions("T",298.15);
			BOOST_REQUIRE_EQUAL(calculate("T"), 298.15);
		}
	BOOST_AUTO_TEST_SUITE_END()

	BOOST_AUTO_TEST_SUITE(MathParserComplexOperations)
		BOOST_AUTO_TEST_CASE(MultiplicationWithParentheses)
		{
			BOOST_REQUIRE_EQUAL(calculate("(6+4)*(30/5)"), 60);
		}
	BOOST_AUTO_TEST_SUITE_END()

	BOOST_AUTO_TEST_SUITE(MathParserExceptionHandling)
		BOOST_AUTO_TEST_CASE(DivisionByZero)
		{
			BOOST_REQUIRE_THROW(calculate("1/0"), divide_by_zero_error);
		}
	BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
