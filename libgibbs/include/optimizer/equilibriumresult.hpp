/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// declaration for EquilibriumResult

#ifndef INCLUDED_EQUILIBRIUMRESULT
#define INCLUDED_EQUILIBRIUMRESULT

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "libgibbs/include/compositionset.hpp"
#include "libgibbs/include/conditions.hpp"

namespace Optimizer {
template<typename T = double> struct Component {
	T site_fraction; // Site fraction
	T chemical_potential; // Chemical potential (relative to reference state)
};
template<typename T = double> struct Sublattice {
	T sitecount; // Number of sites
	std::map<std::string, Component<T> > components; // Components in sublattice
};
template<typename T = double> struct Phase {
	T f; // Phase fraction
	Optimizer::PhaseStatus status; // Phase status
	T chemical_potential(const std::string &) const; // Chemical potential of species in phase
	T mole_fraction(const std::string &) const; //  Mole fraction of species in phase
	T energy(const std::map<std::string,T> &variables, const evalconditions &conditions) const { // Energy of the phase
		return compositionset.evaluate_objective(conditions, variables);
	}
	std::vector<Sublattice<T> > sublattices; // Sublattices in phase
	CompositionSet compositionset; // CompositionSet object (contains model ASTs)

	Phase() : f(0), status(Optimizer::PhaseStatus::SUSPENDED) { }

	// move constructor
	Phase(Phase &&other) :
		f(std::move(other.f)),
		status(std::move(other.status)),
		sublattices(std::move(other.sublattices)),
		compositionset(std::move(other.compositionset)) {
	}
	// move assignment
	Phase & operator= (Phase &&other) {
		this->f = std::move(other.f);
		this->status = std::move(other.status);
		this->sublattices = std::move(other.sublattices);
		this->compositionset = std::move(other.compositionset);
		return *this;
	}

	Phase(const Phase &) = delete;
	Phase & operator=(const Phase &) = delete;
};

// GibbsOpt will fill the EquilibriumResult structure when finalize_solution() is called
// Note: By shifting some things around, it would probably be possible to move all of EquilibriumResult into Equilibrium.
// Only GibbsOpt and Equilibrium will have direct access to a filled EquilibriumResult, and both will control access to it.
// This justifies use of public data members.
template<typename T = double> struct EquilibriumResult {
public:
	typedef std::map<std::string, Phase<T> > PhaseMap;
	typedef std::map<std::string, T> VariableMap;
	double walltime; // Wall clock time to perform calculation
	int itercount; // Number of iterations to perform calculation
	T N; // Total system size in moles (TODO: should eventually be a fixed variable accessed by variables["N"])
	PhaseMap phases; // Phases in equilibrium
	VariableMap variables; // optimized values of all variables
	// TODO: One day all state variables should be fixed variables in the optimization, and evalconditions should go away
	evalconditions conditions; // conditions object for the Equilibrium

	T chemical_potential(const std::string &) const; // Chemical potentials of all entered species
	T mole_fraction(const std::string &) const; //  Mole fraction of species in equilibrium
	T energy() const { // Energy of the system
		T retval = 0;
		for (auto i = phases.begin(); i != phases.end(); ++i) {
			retval += i->second.f * i->second.energy(variables, conditions);
		}
		return retval;
	}

	EquilibriumResult() {}

	EquilibriumResult(EquilibriumResult &&other) :
		walltime(other.walltime),
		itercount(other.itercount),
		N(other.N),
		phases(std::move(other.phases)) {
	}

	EquilibriumResult & operator= (EquilibriumResult &&other) {
		this->walltime = other.walltime;
		this->itercount = other.itercount;
		this->N = other.N;
		this->phases = std::move(other.phases);
		return *this;
	}

	EquilibriumResult(const EquilibriumResult &) = delete;
	EquilibriumResult & operator=(const EquilibriumResult &) = delete;
};
}

#endif
