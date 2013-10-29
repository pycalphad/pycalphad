/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// header file for Constraint class and enums

#ifndef INCLUDED_CONSTRAINT
#define INCLUDED_CONSTRAINT

#include "libtdb/include/structure.hpp"
#include <string>
#include <boost/spirit/include/support_utree.hpp>

boost::spirit::utree mole_fraction(
	const std::string &phase_name,
	const std::string &spec_name,
	const Sublattice_Collection::const_iterator ref_subl_iter_start,
	const Sublattice_Collection::const_iterator ref_subl_iter_end
	);

enum class ConstraintOperatorType {
	EQUALITY_CONSTRAINT
};

class Constraint {
public:
	boost::spirit::utree lhs;
	boost::spirit::utree rhs;
	ConstraintOperatorType op;
	std::string name;
	virtual ~Constraint() {};
};

class MassBalanceConstraint : public Constraint {
public:
	typedef Phase_Collection::const_iterator PhaseIterator;
	MassBalanceConstraint(
			const PhaseIterator phase_begin,
			const PhaseIterator phase_end,
			const std::string &spec_name,
			const double &moles);
};

class SublatticeBalanceConstraint : public Constraint {
public:
	typedef Sublattice_Collection::const_iterator SublatticeIterator;
	SublatticeBalanceConstraint(
			const std::string &phase_name,
			const int &sublindex,
			const std::vector<std::string>::const_iterator spec_begin,
			const std::vector<std::string>::const_iterator spec_end);
};

class PhaseFractionBalanceConstraint : public Constraint {
public:
	typedef Phase_Collection::const_iterator PhaseIterator;
	PhaseFractionBalanceConstraint(const PhaseIterator phase_begin, const PhaseIterator phase_end);
};

class StateVariableConstraint : public Constraint {

};


// perhaps a ConstraintManager object handles the active phases and components along with all constraints
// this object would convert all of the constraints into a form the optimizer engine can handle
// this object would have add and remove methods for constraints
// ConstraintManager would be how Equilibrium objects handle constraints

class ConstraintManager {
protected:
	std::vector<Constraint> constraints;
public:
	void addConstraint(Constraint cons);
	void deleteConstraint();

	typedef std::vector<Constraint>::const_iterator ConstraintIterator;
	ConstraintIterator begin();
	ConstraintIterator end();
};

#endif
