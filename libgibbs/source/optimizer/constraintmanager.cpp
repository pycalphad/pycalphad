/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definitions for ConstraintManager class

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/constraint.hpp"

void ConstraintManager::addConstraint(Constraint cons) {
	// Maybe I need to do error checking here, but upstream can construct cons if they want and check it separately
	constraints.push_back(cons);
}

void ConstraintManager::deleteConstraint() {
	// TODO: make deletion of constraints work
	return;
}
