// param_grammar.h -- grammar declaration for PARAMETER commands

#ifndef INCLUDED_PARAM_GRAMMAR
#define INCLUDED_PARAM_GRAMMAR

#include "libtdb/include/warning_disable.hpp"
#include "libtdb/include/structure.hpp"
#include "libtdb/include/grammars/function_grammar.hpp"

typedef std::vector<std::vector<std::string>> NestedVector;

struct param_grammar :
	boost::spirit::qi::grammar<std::string::const_iterator, Parameter(), boost::spirit::ascii::space_type>
{

	param_grammar(const boost::spirit::qi::symbols<char, boost::spirit::utree>& functions, 
		const boost::spirit::qi::symbols<char, boost::spirit::utree>& variables, const Species_Collection& myspecies);
	boost::spirit::qi::rule<std::string::const_iterator, Parameter(), boost::spirit::ascii::space_type> start;
	boost::spirit::qi::rule<std::string::const_iterator, std::string(), boost::spirit::ascii::space_type> identifier;
	boost::spirit::qi::rule<std::string::const_iterator, std::string(), boost::spirit::ascii::space_type> speciesname;
	boost::spirit::qi::rule<std::string::const_iterator, std::string(), boost::spirit::ascii::space_type> species;
	boost::spirit::qi::rule<std::string::const_iterator, std::vector<std::string>(), boost::spirit::ascii::space_type> sublattice;
	boost::spirit::qi::rule<std::string::const_iterator, NestedVector(), boost::spirit::ascii::space_type> 
		constituent_array;
	function_grammar func_expr; // defined in function_grammar.cpp
};

#endif