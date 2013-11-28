/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// math_expr.hpp -- header for FORTRAN-like mathematical expressions parser

#ifndef INCLUDED_MATH_EXPR
#define INCLUDED_MATH_EXPR

#include "libtdb/include/warning_disable.hpp"
#include "libgibbs/include/conditions.hpp"
#include <string>
#include <boost/spirit/include/support_utree.hpp>

boost::spirit::utree const process_utree(
		boost::spirit::utree const&,
		evalconditions const&,
		std::map<std::string, int> const&,
		double* const
		);
boost::spirit::utree const process_utree(
		boost::spirit::utree const&,
		evalconditions const&
		);
boost::spirit::utree const simplify_utree(boost::spirit::utree const& ut);
boost::spirit::utree const differentiate_utree(boost::spirit::utree const& ut, std::string const& diffvar);
template <typename T> bool is_allowed_value(T &);
bool is_zero_tree(const boost::spirit::utree &ut);

#endif
