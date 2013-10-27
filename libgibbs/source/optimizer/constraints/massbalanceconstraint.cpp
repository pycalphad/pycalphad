/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition of MassBalanceConstraint class

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/constraint.hpp"
#include "libtdb/include/structure.hpp"

using boost::spirit::utree;
typedef boost::spirit::utree_type utree_type;

MassBalanceConstraint::MassBalanceConstraint(
		const PhaseIterator phase_begin,
		const PhaseIterator phase_end,
		const std::string &spec_name,
		const double &moles
		) {
	utree ret_tree;
	op = ConstraintOperatorType::EQUALITY_CONSTRAINT;
	name = spec_name + " Mass Balance";
	for (auto phase_iter = phase_begin; phase_iter != phase_end; ++phase_iter) {
		utree phase_tree;
		phase_tree.push_back("*");
		phase_tree.push_back(phase_iter->first + "_FRAC");
		phase_tree.push_back(
				mole_fraction(
						phase_iter->first,
						spec_name,
						phase_iter->second.get_sublattice_iterator(),
						phase_iter->second.get_sublattice_iterator_end()
						)
					);
		if (ret_tree.which() == utree_type::nil_type) ret_tree.swap(phase_tree);
		else {
			utree temp_tree;
			temp_tree.push_back("+");
			temp_tree.push_back(ret_tree);
			temp_tree.push_back(phase_tree);
			ret_tree.swap(temp_tree);
		}
	}
	lhs = ret_tree;
	rhs = utree(moles);

}
