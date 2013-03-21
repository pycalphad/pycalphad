/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// evaluate.h -- header file for Gibbs energy minimization function
// This file is for use outside the optimizer engine

#ifndef INCLUDED_EVALUATE
#define INCLUDED_EVALUATE

class Database;
struct evalconditions;

void evaluate(const Database &DB, const evalconditions &conditions);

#endif
