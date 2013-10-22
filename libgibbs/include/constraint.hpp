/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// header file for Constraint class and enums

#include <boost/spirit/include/support_utree.hpp>

enum class ConstraintOperatorType {
	EQUALITY_CONSTRAINT
};

class Constraint {
protected:
	boost::spirit::utree lhs;
	boost::spirit::utree rhs;
	ConstraintOperatorType op;
public:
	void evaluate_constraint();
	virtual ~Constraint() {};
};

class MassBalanceConstraint : public Constraint {

};

class PhaseFractionBalanceConstraint : public Constraint {

};

class StateVariableConstraint : public Constraint {

};
