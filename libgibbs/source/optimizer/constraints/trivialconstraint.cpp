/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition of TrivialConstraint class

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/constraint.hpp"

using boost::spirit::utree;
typedef boost::spirit::utree_type utree_type;

TrivialConstraint::TrivialConstraint() {
	op = ConstraintOperatorType::EQUALITY_CONSTRAINT;
	name = "Trivial Constraint";
	lhs = utree(0);
	rhs = utree(0);
}
