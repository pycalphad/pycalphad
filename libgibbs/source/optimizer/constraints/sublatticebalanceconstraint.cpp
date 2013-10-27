/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition of SublatticeBalanceConstraint class

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/constraint.hpp"
#include <sstream>

using boost::spirit::utree;
typedef boost::spirit::utree_type utree_type;

SublatticeBalanceConstraint::SublatticeBalanceConstraint(const std::string &phase_name,
		const int &sublindex,
		const std::vector<std::string>::const_iterator spec_begin,
		const std::vector<std::string>::const_iterator spec_end) {
	op = ConstraintOperatorType::EQUALITY_CONSTRAINT;
	name = "Sublattice Site Fraction Balance";
	rhs = utree(1); // sublattice site fractions must sum to 1
	if (std::distance(spec_begin,spec_end) == 0) {
		// no-phase case
		// this doesn't actually make physical sense but let's just make the constraint trivial
		lhs = utree(1);
		return;
	}
	for (auto i = spec_begin; i != spec_end; ++i) {
		std::stringstream ss;
		ss << phase_name << "_" << sublindex << "_" << *i;
		if (lhs.which() == utree_type::nil_type) {
			// first site fraction is handled differently
			lhs = utree(ss.str());
			continue;
		}
		// add the phase fraction to the summation
		utree temp_tree;
		temp_tree.push_back("+");
		temp_tree.push_back(ss.str());
		temp_tree.push_back(lhs);
		lhs.swap(temp_tree);
	}
}

