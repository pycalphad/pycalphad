// math_expr.h -- header for FORTRAN-like mathematical expressions parser

#ifndef INCLUDED_MATH_EXPR
#define INCLUDED_MATH_EXPR

#include "libtdb/include/warning_disable.hpp"
#include "libtdb/include/conditions.hpp"
#include <boost/spirit/include/support_utree.hpp>

boost::spirit::utree const process_utree(boost::spirit::utree const&, evalconditions const&);

#endif