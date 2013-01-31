// math_expr.h -- header for FORTRAN-like mathematical expressions parser

#ifndef INCLUDED_MATH_EXPR
#define INCLUDED_MATH_EXPR

#include "warning_disable.h"
#include "conditions.h"
#include <boost/spirit/include/support_utree.hpp>

boost::spirit::utree const process_utree(boost::spirit::utree const&, evalconditions const&);

#define SI_GAS_CONSTANT 8.3144621 // J/mol-K

#endif