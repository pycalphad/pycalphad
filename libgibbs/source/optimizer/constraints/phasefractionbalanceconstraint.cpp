/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition of PhaseFractionBalanceConstraint class

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/constraint.hpp"

using boost::spirit::utree;

PhaseFractionBalanceConstraint::PhaseFractionBalanceConstraint(PhaseIterator phase_begin, PhaseIterator phase_end) {
	op = ConstraintOperatorType::EQUALITY_CONSTRAINT;
	name = "Phase Fraction Balance";
	rhs = utree(1); // phase fractions must sum to 1
	if (std::distance(phase_begin,phase_end) == 0) {
		// no-phase case
		// this doesn't actually make physical sense but let's just make the constraint trivial
		lhs = utree(1);
		return;
	}
	for (auto i = phase_begin; i != phase_end; ++i) {
		if (i == phase_begin) {
			// first phase fraction is handled differently
			lhs = utree(phase_begin->first + "_FRAC");
			continue;
		}
		// add the phase fraction to the summation
		utree temp_tree;
		temp_tree.push_back("+");
		temp_tree.push_back(i->first + "_FRAC");
		temp_tree.push_back(lhs);
		lhs.swap(temp_tree);
	}
}
