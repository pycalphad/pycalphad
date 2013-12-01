/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition for EquilibriumResult and associated objects

#include "libgibbs/include/optimizer/equilibriumresult.hpp"

template <typename T> T Optimizer::EquilibriumResult<T>::energy() const {
	T retval = 0;
	for (auto i = phases.begin(); i != phases.end(); ++i) {
		retval += i->f * i->energy(variables, conditions);
	}
	return retval;
}
template <typename T> T Optimizer::Phase<T>::energy(
		const std::map<std::string,T> &variables,
		const evalconditions &conditions) const {
	T retval = 0;
	return retval;
}
