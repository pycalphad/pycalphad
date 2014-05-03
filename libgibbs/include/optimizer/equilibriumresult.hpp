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
#include "libtdb/include/logging.hpp"
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <limits>
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

protected:
    bool species_occupies_all_sublattices(const std::string &name) const {
        // Check if this component is in every sublattice.
        // Alternatively, check if it's in every sublattice except ones only occupied by vacancies. 
        // If either of these conditions hold, we can directly calculate the chemical potential.
        for (auto subl = sublattices.begin(); subl != sublattices.end(); ++subl) {
            // Vacancy-only sublattices don't count 
            if (subl->components.size() == 1 && subl->components.begin()->first == "VA") continue;
            const auto species_find = subl->components.find(name);
            // found in sublattice, keep going
            if (species_find != subl->components.end()) continue;
            // If we get here, we can't define the chemical potential directly
            return false;
            }
        return true;
    }

public:
    T f; // Phase fraction
    Optimizer::PhaseStatus status; // Phase status
    std::vector<Sublattice<T> > sublattices; // Sublattices in phase
    CompositionSet compositionset; // CompositionSet object (contains model ASTs)
    mutable logger pot_log;
    T mole_fraction(const std::string &) const; //  Mole fraction of species in phase
    T energy(const std::map<std::string,T> &variables, const evalconditions &conditions) const { // Energy of the phase
            return compositionset.evaluate_objective(conditions, variables);
    }
    T chemical_potential(const std::string &name, const std::map<std::string,T> &variables, const evalconditions &conditions) const {
            // Chemical potential of species in phase
            // Based on Hillert, 2008, p. 77-78
            // NOTE: It is NOT guaranteed that this function is well-defined for all phases/components.
            // Some phases,  e.g.,  line compounds, only have well-defined chemical potentials at
            // multi-phase equilibrium,  so they will have to be calculated together.
            BOOST_LOG_NAMED_SCOPE("Phase::chemical_potential");
            BOOST_LOG_CHANNEL_SEV(pot_log, "optimizer", debug) << "mu " << name << " in " << compositionset.name();
            if (name == "VA") return 0; // chemical potential of vacancy is defined to be zero
            T ret_potential;
            std::size_t total_site_count = 0;
            const bool species_occupies_all_sublattices = this->species_occupies_all_sublattices(name);
            std::map<int,T> gradient = compositionset.evaluate_single_phase_objective_gradient(conditions, variables);
            ret_potential = energy(variables, conditions); // First term is the energy of the phase
            BOOST_LOG_SEV(pot_log, debug) << "energy = " << ret_potential;
            
            for (auto subl = sublattices.begin(); subl != sublattices.end(); ++subl) {
                // Vacancy-only sublattices don't count 
                if (subl->components.size() == 1 && subl->components.begin()->first == "VA") continue;
                total_site_count += subl->sitecount;
            }
            
            if (species_occupies_all_sublattices) {
                for (auto subl = sublattices.begin(); subl != sublattices.end(); ++subl) {
                    T site_count = subl->sitecount;
                    // Iterate over all components
                    for (auto j = subl->components.begin(); j != subl->components.end(); ++j) {
                        if (j->first == "VA") continue;     //  skip vacancies
                        std::stringstream sitefraction_name;
                        sitefraction_name << compositionset.name() << "_" << std::distance(sublattices.begin(),subl) << "_" << j->first;
                        const auto component_find = variables.find(sitefraction_name.str());
                        const auto gradient_component_index = std::distance(variables.begin(), component_find);
                        const T gradient_value = gradient.at(gradient_component_index);
                        ret_potential -= component_find->second * gradient_value;
                        
                        if (j->first == name) {
                            // There's an extra term associated with this being the species of interest
                            ret_potential += gradient_value;
                        }
                    }
                }
            }
            else {
                auto ref_component_find = conditions.elements.begin();
                while (ref_component_find != conditions.elements.end() && 
                    !this->species_occupies_all_sublattices(*(ref_component_find))) {
                    ref_component_find++;
                }
                //  ref_component_find points to a chemical potential which can be calculated
                //  or it points to nothing
                if (ref_component_find == conditions.elements.end()) return std::numeric_limits<T>::quiet_NaN();
                const T ref_chemical_potential = this->chemical_potential(*ref_component_find,  variables,  conditions);
                //  Now we need to locate a sublattice containing the reference component and the current one
                auto sublattice_find = sublattices.end();
                for (auto subl = sublattices.begin(); subl != sublattices.end(); ++subl) {
                    const auto component_find = subl->components.find(name);
                    if (component_find != subl->components.end()) {
                        sublattice_find = subl;
                        break;
                    }
                }
                if (sublattice_find != sublattices.end()) {
                    // The chemical potential can be calculated based on the reference component
                    std::stringstream ref_sitefraction_name;
                    ref_sitefraction_name << compositionset.name() << "_" << std::distance(sublattices.begin(),sublattice_find) << "_" << *ref_component_find;
                    const auto ref_component_find = variables.find(ref_sitefraction_name.str());
                    const auto ref_gradient_component_index = std::distance(variables.begin(), ref_component_find);
                    const T ref_gradient_value = gradient.at(ref_gradient_component_index);
                    
                    std::stringstream component_sitefraction_name;
                    component_sitefraction_name << compositionset.name() << "_" << std::distance(sublattices.begin(),sublattice_find) << "_" << name;
                    const auto component_find = variables.find(component_sitefraction_name.str());
                    const auto gradient_component_index = std::distance(variables.begin(), component_find);
                    const T component_gradient_value = gradient.at(gradient_component_index);
                    
                    const auto site_count = sublattice_find->sitecount;
                    
                    ret_potential = (total_site_count/site_count) * (component_gradient_value - ref_gradient_value) + ref_chemical_potential;
                }
                else return std::numeric_limits<T>::quiet_NaN();
            }

            BOOST_LOG_SEV(pot_log, debug) << "ret_potential = " << ret_potential;
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
