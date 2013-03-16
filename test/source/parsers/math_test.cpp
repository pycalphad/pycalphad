// math_test.cpp -- test suite for the arithmetic parser

#include "test/include/test_pch.hpp"
#include "libtdb/include/warning_disable.hpp"
#include "test/include/fixtures/fixture_math.hpp"
#include <string>
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>


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
			BOOST_REQUIRE_EQUAL(calculate("7*-3"), -21);
		}
		BOOST_AUTO_TEST_CASE(Division)
		{
			BOOST_REQUIRE_EQUAL(calculate("400/2"), 200);
		}
		BOOST_AUTO_TEST_CASE(ScientificNotation)
		{
			BOOST_REQUIRE_CLOSE_FRACTION(calculate("1.56E-34"), 1.56e-34, 1e-15);
			BOOST_REQUIRE_CLOSE_FRACTION(calculate("1.56e-34"), 1.56e-34, 1e-15);
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
		BOOST_AUTO_TEST_CASE(LogLabelBehavesLikeNaturalLog)
		{
			BOOST_REQUIRE_CLOSE_FRACTION(calculate("LOG(2.7182818284590451)"), 1, 1e-15);
			BOOST_REQUIRE_CLOSE_FRACTION(calculate("LoG(2.7182818284590451)"), 1, 1e-15);
		}
		BOOST_AUTO_TEST_CASE(Exponentiation)
		{
			BOOST_REQUIRE_EQUAL(calculate("13**7"), 62748517);
			BOOST_REQUIRE_EQUAL(calculate("4**0.5"), 2);
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
			BOOST_REQUIRE_THROW(calculate("t"), syntax_error);
			set_conditions("P",101325);
			BOOST_REQUIRE_EQUAL(calculate("P"), 101325);
			BOOST_REQUIRE_THROW(calculate("p"), syntax_error);
		}
		BOOST_AUTO_TEST_CASE(StateVariableOperations)
		{
			clear_conditions();
			set_conditions("T",300);
			BOOST_REQUIRE_EQUAL(calculate("T+20"), 320);
			BOOST_REQUIRE_EQUAL(calculate("T-10"), 290);
			BOOST_REQUIRE_EQUAL(calculate("20+T"), 320);
			BOOST_REQUIRE_EQUAL(calculate("-10+T"), 290);
			BOOST_REQUIRE_EQUAL(calculate("-T+400"), 100);
			BOOST_REQUIRE_EQUAL(calculate("T/5"), 60);
			BOOST_REQUIRE_EQUAL(calculate("T*2"), 600);
			BOOST_REQUIRE_EQUAL(calculate("(T/100)**2"), 9);
			BOOST_REQUIRE_CLOSE_FRACTION(calculate("exp(T/100)"), 20.085536923187668, 1e-15);
			BOOST_REQUIRE_CLOSE_FRACTION(calculate("ln(T)"), 5.7037824746562009, 1e-15);
		}
		BOOST_AUTO_TEST_CASE(WhiteSpaceHandling)
		{
			BOOST_REQUIRE_EQUAL(calculate("23                +               3"), 26);
			BOOST_REQUIRE_THROW(calculate("2 3           + 7"), syntax_error);
			BOOST_REQUIRE_EQUAL(calculate("       50       **        0"), 1);
			BOOST_REQUIRE_THROW(calculate("3 * * 2"), syntax_error);
		}
		BOOST_AUTO_TEST_CASE(TabHandling)
		{
			BOOST_REQUIRE_EQUAL(calculate("\t4\t+\t    5"), 9);
		}
		BOOST_AUTO_TEST_CASE(MathParserNegationAfterNewline)
		{
			clear_conditions();
			set_conditions("T", 2);
			BOOST_REQUIRE_EQUAL(calculate("100\n-100*T"), -100);
		}
	BOOST_AUTO_TEST_SUITE_END()

	BOOST_AUTO_TEST_SUITE(MathParserComplexOperations)
		BOOST_AUTO_TEST_CASE(MultiplicationWithParentheses)
		{
			BOOST_REQUIRE_EQUAL(calculate("(6+4)*(30/5)"), 60);
		}
		BOOST_AUTO_TEST_CASE(NestedOperations)
		{
			clear_conditions();
			set_conditions("T", 300);
			BOOST_REQUIRE_CLOSE_FRACTION(calculate("(2*6*ln(4*T**2))"), 153.5263117251875, 1e-15);
			BOOST_REQUIRE_CLOSE_FRACTION(
					calculate("-7285.889+119.139857*T-23.7592624*T*LN(T)      -.002623033*T**2+1.70109E-07*T**3-3293*T**(-1)"),
					-12441.68794003007589412,
					1e-15
					);
		}
	BOOST_AUTO_TEST_SUITE_END()

	BOOST_AUTO_TEST_SUITE(MathParserThrows)
		BOOST_AUTO_TEST_CASE(DivisionByZero)
		{
			BOOST_REQUIRE_THROW(calculate("1/0"), divide_by_zero_error);
			clear_conditions();
			set_conditions("T",0);
			BOOST_REQUIRE_THROW(calculate("1/T"), divide_by_zero_error);
		}
		BOOST_AUTO_TEST_CASE(FloatingPointClassification)
		{
			BOOST_REQUIRE_THROW(calculate("10**10**10**10"), floating_point_error);
			BOOST_REQUIRE_THROW(calculate("-10**10**10**10"), floating_point_error);
			BOOST_REQUIRE_THROW(calculate("-10**10**10**10*0"), floating_point_error);
			BOOST_REQUIRE_THROW(calculate("(-10**10**10**10)**0"), floating_point_error);
			BOOST_REQUIRE_THROW(calculate("0*-10**10**10**10"), floating_point_error);
			BOOST_REQUIRE_THROW(calculate("0**-10**10**10**10"), floating_point_error);
		}
		BOOST_AUTO_TEST_CASE(OutsideFunctionDomain)
		{
			BOOST_REQUIRE_THROW(calculate("ln(-1)"), domain_error);
			BOOST_REQUIRE_THROW(calculate("(-4)**0.5"), domain_error);
			BOOST_REQUIRE_THROW(calculate("-4**0.5"), domain_error);
		}
	BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
