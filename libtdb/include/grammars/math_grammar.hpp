// math_grammar.h -- grammar declaration for FORTRAN-like arithmetic expressions
// Based on http://www.boost.org/doc/libs/1_51_0/libs/spirit/example/qi/calc_utree_ast.cpp

/*=============================================================================
    Copyright (c) 2001-2011 Hartmut Kaiser
    Copyright (c) 2001-2011 Joel de Guzman
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#ifndef INCLUDED_MATH_GRAMMAR
#define INCLUDED_MATH_GRAMMAR

#include "libtdb/include/warning_disable.hpp"
#include <boost/spirit/include/support_utree.hpp>
#include <boost/spirit/include/qi.hpp>

struct calculator: boost::spirit::qi::grammar<std::string::const_iterator, boost::spirit::ascii::space_type, boost::spirit::utree()> {
	calculator(const boost::spirit::qi::symbols<char, boost::spirit::utree>& functions, 
		const boost::spirit::qi::symbols<char, boost::spirit::utree>& variables);
	boost::spirit::qi::rule<std::string::const_iterator, boost::spirit::ascii::space_type, boost::spirit::utree()> expression, term, factor, function_name;
};

#endif
