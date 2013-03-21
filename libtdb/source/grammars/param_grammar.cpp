/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// param_grammar.cpp -- grammar definition for PARAMETER commands

//#define BOOST_SPIRIT_DEBUG

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/grammars/param_grammar.hpp"

#include <boost/spirit/include/support_utree.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_function.hpp>

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace spirit = boost::spirit;

struct findSpecies_
{
	template <typename T1, typename T2, typename T3>
	struct result { typedef void type; };

	void operator()(Species &output, const std::string &input, const Species_Collection& myspecies) const
	{
			auto spec_iter = myspecies.find(input);
			if (spec_iter != myspecies.end()) output = spec_iter->second;
			else {
				// TODO: throw a 'Species missing' exception
				BOOST_THROW_EXCEPTION(parse_error() << specific_errinfo("Unknown species: " + input));
			}
	}
};
boost::phoenix::function<findSpecies_> const findSpecies;

///////////////////////////////////////////
// Grammar for PARAMETERs
///////////////////////////////////////////
// Grammar takes form: <identifier>(<phase>_<suffix>, <constituent array>; <digit>) <func_expr>
param_grammar::param_grammar(const boost::spirit::qi::symbols<char, boost::spirit::utree>& functions, 
	const boost::spirit::qi::symbols<char, spirit::utree>& variables, const Species_Collection& myspecies) : 
qi::grammar<std::string::const_iterator, Parameter(), ascii::space_type>::base_type(start), func_expr(functions,variables) {
	using qi::lit;
	using qi::lexeme;
	using qi::_val;
	using qi::_1;
	using qi::double_;
	using qi::int_;
	using qi::attr;
	using ascii::char_;
	using ascii::alnum;
	using ascii::string;
	using namespace qi::labels;

	identifier = lexeme[+alnum - char_("(,_")]; // identifier for the parameter: G, L, TC, etc. or the phase name (same match rule)

	speciesname = lexeme[+(alnum | char_('_') | char_('*')) - char_(":,;")];

	species = speciesname; //[findSpecies(_val,_1,myspecies)];

	sublattice = (species % char_(','));

	constituent_array = (sublattice % char_(':'));
	//constituent_array = +(alnum | char_('_'));

	start = identifier // parameter type
		>> lit('(')
		>> identifier // first part of phase name
		>> ((lit('_') >> identifier) | attr("")) // optional second part of phase name
		>> lit(',')
		>> constituent_array
		>> ((lit(';') >> int_) | attr(0)) // if no digit is given, emit 0 as digit by default
		>> lit(')')
		>> func_expr;

	BOOST_SPIRIT_DEBUG_NODE(identifier);
	BOOST_SPIRIT_DEBUG_NODE(speciesname);
}
