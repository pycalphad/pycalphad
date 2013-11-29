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
	T energy() const; // Energy of the phase
	std::vector<Sublattice<T> > sublattices; // Sublattices in phase
	std::unique_ptr<CompositionSet> compositionset; // pointer to CompositionSet object (contains model ASTs)

	Phase() : f(0), status(Optimizer::PhaseStatus::SUSPENDED) { };

	// move constructor
	Phase(Phase &&other) :
		f(other.f), status(other.status), sublattices(other.sublattices), compositionset(std::move(other.compositionset)) {
	}
	// move assignment
	Phase & operator= (Phase &&other) {
		this->f = other.f;
		this->status = other.status;
		this->sublattices = std::move(other.sublattices);
		this->compositionset = std::move(other.compositionset);
		return *this;
	}

	Phase(const Phase &) = delete;
	Phase & operator=(const Phase &) = delete;
};

// GibbsOpt will fill the EquilibriumResult structure when
template<typename T = double> struct EquilibriumResult {
	double walltime; // Wall clock time to perform calculation
	int itercount; // Number of iterations to perform calculation
	double N; // Total system size in moles
	T chemical_potential(const std::string &) const; // Chemical potentials of all entered species
	T mole_fraction(const std::string &) const; //  Mole fraction of species in equilibrium
	T energy() const; // Energy of the system
	std::map<std::string, std::unique_ptr<Phase<T> > > phases; // Phases in equilibrium

	EquilibriumResult(const EquilibriumResult &) = delete;
	EquilibriumResult & operator=(const EquilibriumResult &) = delete;
};
}

#endif
