/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// function_grammar.cpp -- grammar construction for TDB file T-dependent FUNCTIONs

//#define BOOST_SPIRIT_DEBUG

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/grammars/function_grammar.hpp"

#include <limits>
#include <boost/spirit/include/support_utree.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_function.hpp>

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace spirit = boost::spirit;


struct constraint
{
	template <typename T1, typename T2>
	struct result { typedef void type; };

	constraint(char* var) : var(var) {}

	void operator()(spirit::utree &output, std::vector<boost::variant<double,spirit::utree>> &vec) const
	{
		//std::cout << "constraint in" << std::endl;
		double lowlimit = -std::numeric_limits<double>::max();
		double highlimit = std::numeric_limits<double>::max();
		spirit::utree buildtree;
		for (auto i = vec.begin(); i != vec.end(); ++i) {
			//std::cout << "vec element type: " << (*i).which() << std::endl;
			if ((lowlimit == -std::numeric_limits<double>::max()) && ((*i).type() == typeid(double))) {
				// lowlimit is not yet set, we should see that first
				//std::cout << "setting lowlimit ";
				lowlimit = boost::get<double>(*i);
				//std::cout << "to " << lowlimit << std::endl;
			}
			else {
				// lowlimit is set, we should see either our utree or another double
				if ((*i).type() == typeid(spirit::utree)) {
					//std::cout << "this is a utree" << std::endl;
					spirit::utree ast = boost::get<spirit::utree>(*i);
					buildtree.push_back("@");
					buildtree.push_back(var);
					buildtree.push_back(lowlimit);
					++i;
					if (i!=vec.end()) {
						//std::cout << "setting highlimit" << std::endl;
						if ((*i).type() == typeid(double)) highlimit = boost::get<double>(*i);
					}
					//std::cout << "lowlimit: " << lowlimit << std::endl;
					//std::cout << "highlimit: " << highlimit << std::endl;
					buildtree.push_back(highlimit);
					buildtree.push_back(ast);
					lowlimit = highlimit; // the old highlimit is the next element's lowlimit
					if (highlimit == std::numeric_limits<double>::max()) break; // we're at the end of the sequence (final highlimit not set)
					else highlimit = std::numeric_limits<double>::max();
				}
				else if ((*i).type() == typeid(double)) {
				}
				else {
					// TODO: throw an exception here
				}
			}
			//std::cout << "lowlimit on endloop: " << lowlimit << std::endl;
		}
		output.swap(buildtree);
		buildtree.clear();
		//std::cout << "constraint out" << std::endl;
	}

	char const* var;
};
boost::phoenix::function<constraint> Trange = constraint("T");

///////////////////////////////////////////
// Grammar for FUNCTIONs
///////////////////////////////////////////
function_grammar::function_grammar(const boost::spirit::qi::symbols<char, boost::spirit::utree>& functions, 
		const boost::spirit::qi::symbols<char, spirit::utree>& variables) : 
qi::grammar<std::string::const_iterator, spirit::utree(), ascii::space_type>::base_type(start), mycalc(functions,variables) {
	using qi::lit;
	using qi::lexeme;
	using qi::_val;
	using qi::_1;
	using qi::_2;
	using qi::_3;
	using qi::eps;
	using qi::attr;
	using qi::_a;
	using qi::double_;
	using ascii::char_;
	using ascii::alnum;
	using ascii::string;
	using namespace qi::labels;

	//text = lexeme[+(char_ - ' ')        [_val += _1]]; // save function name

	randomtext = *(char_);

	mathexpr = ((mycalc - char_(';')) >> lit(';')); // could just be a constant (double)

	firstexpression = (lexeme[double_ >> lit(' ')] >> mathexpr >> (double_ | attr(std::numeric_limits<double>::max())));
	firstexpnomax = (lexeme[double_ >> lit(' ')] >> mathexpr);

	expression = (mathexpr >> double_);
	lastexpnomax = (mathexpr);

	expressions = (firstexpression >> *(lit('Y') >> expression) >> -(lit('Y') >> lastexpnomax))
		| firstexpnomax
		| (firstexpression >> lit('Y') >> lastexpnomax)
		| firstexpression;

	expressiontree = expressions[Trange(_val, _1)];

	endstring = *lit(',') >> lit('N') >> -(lit("REF") >> -lit(':')) >> randomtext;

	start = (expressiontree)[_val = _1] >> endstring; // TODO: support for REF command

	//BOOST_SPIRIT_DEBUG_NODE(text);
	BOOST_SPIRIT_DEBUG_NODE(expressions);
	BOOST_SPIRIT_DEBUG_NODE(start);
	BOOST_SPIRIT_DEBUG_NODE(endstring);
	BOOST_SPIRIT_DEBUG_NODE(firstexpression);
	BOOST_SPIRIT_DEBUG_NODE(firstexpnomax);
	BOOST_SPIRIT_DEBUG_NODE(lastexpnomax);
	BOOST_SPIRIT_DEBUG_NODE(expression);
	BOOST_SPIRIT_DEBUG_NODE(mathexpr);
}
