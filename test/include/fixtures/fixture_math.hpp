// fixture_math.hpp -- header file for math parser text fixture

#ifndef FIXTURE_MATH_HPP_
#define FIXTURE_MATH_HPP_

#include "libtdb/include/exceptions.hpp"
#include "libtdb/include/grammars/math_grammar.hpp"
#include "libtdb/include/utils/math_expr.hpp"
#include "libtdb/include/conditions.hpp"
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
		return 0; // impossible
	}
	boost::spirit::qi::symbols<char, boost::spirit::utree> macros; // all of the macros (FUNCTIONs in Thermo-Calc lingo)
	boost::spirit::qi::symbols<char, boost::spirit::utree> statevars; // all valid state variables
	calculator calc_parser;
	evalconditions conditions;
};


#endif /* FIXTURE_MATH_HPP_ */
