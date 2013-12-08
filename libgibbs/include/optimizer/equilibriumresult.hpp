/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// declaration for EquilibriumResult

#ifndef INCLUDED_EQUILIBRIUMRESULT
#define INCLUDED_EQUILIBRIUMRESULT

#include "libgibbs/include/compositionset.hpp"
#include "libgibbs/include/conditions.hpp"
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <sstream>

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
	std::vector<Sublattice<T> > sublattices; // Sublattices in phase
	CompositionSet compositionset; // CompositionSet object (contains model ASTs)
	T mole_fraction(const std::string &) const; //  Mole fraction of species in phase
	T energy(const std::map<std::string,T> &variables, const evalconditions &conditions) const { // Energy of the phase
		return compositionset.evaluate_objective(conditions, variables);
	}
	T chemical_potential(const std::string &name, const std::map<std::string,T> &variables, const evalconditions &conditions) const {
		// Chemical potential of species in phase
		// The expression for this in terms of site fractions involves a summation over all sublattices
		// mu_k = G(...) + sum[s]((1/(b_s))*(dG/dy[k,s]*sum[v](b_v*(1-y_vVa))) - sum[j](dG/dy[j,s]*sum[v](b_v*y[j,v])))
		// For the single-sublattice case, this reduces to the well-known expression for mole fractions derived by Hillert
		T ret_potential;
		T sum_of_sublattice_coefficients = 0;
		std::map<int,T> gradient = compositionset.evaluate_objective_gradient(conditions, variables);
		ret_potential = energy(variables, conditions); // First term is the energy of the phase

		// First sublattice iteration to calculate a constant with respect to iteration
		for (auto subl = sublattices.begin(); subl != sublattices.end(); ++subl) {
			T vacancy_sitefraction = 0;
			std::stringstream vacancy_sitefraction_name;
			vacancy_sitefraction_name << compositionset.name() << "_" << std::distance(sublattices.begin(),subl) << "_VA";
			const auto vacancy_find = variables.find(vacancy_sitefraction_name.str());
			if (vacancy_find != variables.end()) vacancy_sitefraction = vacancy_find->second;
			sum_of_sublattice_coefficients += subl->sitecount * (1 - vacancy_sitefraction);
		}

		// Main sublattice iteration: Iterate over all sublattices in this phase
		for (auto subl = sublattices.begin(); subl != sublattices.end(); ++subl) {
			// Define the variable name corresponding to the site fraction of interest in this sublattice
			std::stringstream primary_sitefraction_name;
			primary_sitefraction_name << compositionset.name() << "_" << std::distance(sublattices.begin(),subl) << "_" << name;

			// Search for it in the variables
			const auto primary_component_find = variables.find(primary_sitefraction_name.str());

			// Does the named component exist in this sublattice?
			if (primary_component_find == variables.end()) continue; // Skip this sublattice, the component doesn't exist here

			// This works because evaluate_objective_gradient() will return the gradient in the same order as "variables"
			const int gradient_variable_index = std::distance(variables.begin(),primary_component_find);
			const T gradient_value = gradient.at(gradient_variable_index);
			T subl_term = gradient_value * sum_of_sublattice_coefficients; // Contribution by named species

			// Now add contribution from all of the other components in this sublattice
			for (auto j = subl->components.begin(); j != subl->components.end(); ++j) {
				if (j->first == name) continue; // exclude the named component
				std::stringstream sitefraction_name;
				sitefraction_name << compositionset.name() << "_" << std::distance(sublattices.begin(),subl) << "_" << j->first;
				const auto component_find = variables.find(sitefraction_name.str());
				const int gradient_component_index = std::distance(variables.begin(), component_find);
				const T other_gradient_value = gradient.at(gradient_component_index);
				T weighted_sum_of_sitefractions = 0;
				for (auto k = sublattices.begin(); k != sublattices.end(); ++k) {
					std::stringstream other_sitefraction_name;
					other_sitefraction_name << compositionset.name() << "_" << std::distance(sublattices.begin(),k) << "_" << j->first;
					const auto other_sublattice_component_find = variables.find(other_sitefraction_name.str());
					if (other_sublattice_component_find == variables.end()) continue; // Component is not in this sublattice
					weighted_sum_of_sitefractions += k->sitecount * other_sublattice_component_find->second;
				}
				subl_term -= other_gradient_value * weighted_sum_of_sitefractions; // Subtract from sublattice contribution
			}

			subl_term = subl_term / subl->sitecount;
			ret_potential += subl_term; // Add sublattice contribution to total chemical potential
		}

		return ret_potential;
	}

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
			const std::string phasefrac(i->first + "_FRAC");
			retval += variables.at(phasefrac) * i->second.energy(variables, conditions);
		}
		return retval;
	}

	EquilibriumResult() {}

	EquilibriumResult(EquilibriumResult &&other) :
		walltime(other.walltime),
		itercount(other.itercount),
		N(other.N),
		phases(std::move(other.phases),
		variables(std::move(other.variables)),
		conditions(std::move(other.conditions))) {
	}

	EquilibriumResult & operator= (EquilibriumResult &&other) {
		this->walltime = other.walltime;
		this->itercount = other.itercount;
		this->N = other.N;
		this->phases = std::move(other.phases);
		this->variables = std::move(other.variables);
		this->conditions = std::move(other.conditions);
		return *this;
	}

	EquilibriumResult(const EquilibriumResult &) = delete;
	EquilibriumResult & operator=(const EquilibriumResult &) = delete;
};
}

#endif
