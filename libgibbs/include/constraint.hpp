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
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>

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


struct jacobian_entry {
	int cons_index; // constraint index (for optimizer)
	int var_index; // differentiating variable index (for optimizer)
	bool trivial; // is it always zero?
	boost::spirit::utree ast; // abstract syntax tree
	jacobian_entry (
			int cons_index_, int var_index_, bool trivial_, boost::spirit::utree ast_) :
				cons_index(cons_index_),
				var_index(var_index_),
				trivial(trivial_),
				ast(ast_) {}
};


struct hessian_entry {
	int cons_index; // constraint/objective index (-1 for objective, for optimizer)
	int var_index1; // differentiating variable index 1 (for optimizer)
	int var_index2; // differentiating variable index 2 (for optimizer)
	boost::spirit::utree ast; // abstract syntax tree
	hessian_entry (
			int cons_index_, int var_index1_, int var_index2_, boost::spirit::utree ast_) :
				cons_index(cons_index_),
				var_index1(var_index1_),
				var_index2(var_index2_),
				ast(ast_) {
		if (var_index2 > var_index1) {
			// Store only the upper triangular form
			int temp = var_index2;
			var_index2 = var_index1;
			var_index1 = temp;
		}
	}
};

typedef boost::multi_index_container<
  hessian_entry,
  boost::multi_index::indexed_by<
    boost::multi_index::ordered_unique<
      boost::multi_index::composite_key<
        hessian_entry,
        BOOST_MULTI_INDEX_MEMBER(hessian_entry,int,var_index1),
        BOOST_MULTI_INDEX_MEMBER(hessian_entry,int,var_index2),
        BOOST_MULTI_INDEX_MEMBER(hessian_entry,int,cons_index)
      >
    >
  >
> hessian_set;

#endif
