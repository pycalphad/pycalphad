/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// evaluate.cpp -- evaluate energies from a Database

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/equilibrium.hpp"
#include "libtdb/include/database.hpp"
#include "libtdb/include/structure.hpp"
#include "libtdb/include/exceptions.hpp"
#include "libgibbs/include/optimizer/optimizer.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include "external/coin/IpIpoptApplication.hpp"
#include "external/coin/IpSolveStatistics.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <boost/io/ios_state.hpp>

using namespace Ipopt;

Equilibrium::Equilibrium(const Database &DB, const evalconditions &conds)
: sourcename(DB.get_info()), conditions(conds) {
	Phase_Collection phase_col;
	for (auto i = DB.get_phase_iterator(); i != DB.get_phase_iterator_end(); ++i) {
		if (conds.phases.find(i->first) != conds.phases.end()) {
			if (conds.phases.at(i->first) == true) phase_col[i->first] = i->second;
		}
	}

	const Phase_Collection::const_iterator phase_iter = phase_col.cbegin();
	const Phase_Collection::const_iterator phase_end = phase_col.cend();

	// TODO: check validity of conditions

	// Create an instance of your nlp...
	SmartPtr<TNLP> mynlp = new GibbsOpt(phase_iter, phase_end, conds);

	// Create an instance of the IpoptApplication
	//
	// We are using the factory, since this allows us to compile this
	// example with an Ipopt Windows DLL
	SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

	app->Options()->SetStringValue("derivative_test","first-order");
	app->Options()->SetStringValue("hessian_approximation","limited-memory");
	//app->Options()->SetNumericValue("bound_relax_factor",1e-7);
	//app->Options()->SetIntegerValue("print_level",12);
	//app->Options()->SetStringValue("derivative_test_print_all","yes");

	// Initialize the IpoptApplication and process the options
	ApplicationReturnStatus status;
	status = app->Initialize();
	if (status != Solve_Succeeded) {
		std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
		BOOST_THROW_EXCEPTION(equilibrium_error() << str_errinfo("Error initializing solver"));
	}

	status = app->OptimizeTNLP(mynlp);

	if (status == Solve_Succeeded ||
			status == Solved_To_Acceptable_Level ||
			status == Search_Direction_Becomes_Too_Small) {
		// Retrieve some statistics about the solve
		Index iter_count = app->Statistics()->IterationCount();
		std::cout << std::endl << std::endl << "*** The problem solved in " << iter_count << " iterations!" << std::endl;

		Number final_obj = app->Statistics()->FinalObjective();
		mingibbs = final_obj * conds.statevars.find('N')->second;
		GibbsOpt* opt_ptr = dynamic_cast<GibbsOpt*> (Ipopt::GetRawPtr(mynlp));
		if (!opt_ptr)
			BOOST_THROW_EXCEPTION(equilibrium_error() << str_errinfo("Internal memory error") << specific_errinfo("dynamic_cast<GibbsOpt*>"));
		ph_map = opt_ptr->get_phase_map();
	}
	else {
		BOOST_THROW_EXCEPTION(equilibrium_error() << str_errinfo("Solver failed to find equilibrium"));
	}
}

std::ostream& operator<< (std::ostream& stream, const Equilibrium& eq) {
	boost::io::ios_flags_saver  ifs( stream ); // preserve original state of the stream once we leave scope
	stream << "Output from LIBGIBBS, equilibrium number = ??" << std::endl;
	stream << "Conditions:" << std::endl;

	// We want the individual phase information to appear AFTER
	// the global system data, but the most efficient ordering
	// requires us to iterate through all phases first.
	// The simple solution is to save the output to a temporary
	// buffer, and then flush it to the output stream later.
	std::stringstream temp_buf;
    temp_buf << std::scientific;

	const auto sv_end = eq.conditions.statevars.cend();
	const auto xf_end = eq.conditions.xfrac.cend();
	for (auto i = eq.conditions.xfrac.cbegin(); i !=xf_end; ++i) {
		stream << "X(" << i->first << ")=" << i->second << ", ";
	}
	for (auto i = eq.conditions.statevars.cbegin(); i != sv_end; ++i) {
		stream << i->first << "=" << i->second;
		// if this isn't the last one, add a comma
		if (std::distance(i,sv_end) != 1) stream << ", ";
	}
	stream << std::endl;
	// should be c + 2 - conditions, where c is the number of components
	size_t dof = eq.conditions.elements.size() + 2 - (eq.conditions.xfrac.size()+1) - eq.conditions.statevars.size();
	stream << "DEGREES OF FREEDOM " << dof << std::endl;

	stream << std::endl;

	double T,P,N;
	T = eq.conditions.statevars.find('T')->second;
	P = eq.conditions.statevars.find('P')->second;
	N = eq.conditions.statevars.find('N')->second;
	stream << "Temperature " << T << " K (" << (T-273.15) << " C), " << "Pressure " << P << " Pa" << std::endl;

	stream << std::scientific; // switch to scientific notation for doubles
    stream << "Number of moles of components " << N << ", Mass ????" << std::endl;
    stream << "Total Gibbs energy " << eq.mingibbs << " Enthalpy ???? " << "Volume ????" << std::endl;

    stream << std::endl;

    const auto ph_end = eq.ph_map.cend();
    // double/double pair is for separate storage of numerator/denominator pieces of fraction
    std::map<std::string,std::pair<double,double>> global_comp;
    for (auto i = eq.ph_map.cbegin(); i != ph_end; ++i) {
    	const auto subl_begin = i->second.second.cbegin();
    	const auto subl_end = i->second.second.cend();
    	const double phasefrac = i->second.first;
    	std::map<std::string,std::pair<double,double>> phase_comp;
    	temp_buf << i->first << "\tStatus ENTERED  Driving force 0" << std::endl; // phase name
    	temp_buf << "Number of moles " << i->second.first * N << ", Mass ???? ";
    	temp_buf << "Mole fractions:" << std::endl;
    	for (auto j = subl_begin; j != subl_end; ++j) {
    		const double stoi_coef = j->first;
    		const double den = stoi_coef;
    		const auto cond_spec_begin = eq.conditions.elements.cbegin();
    		const auto cond_spec_end = eq.conditions.elements.cend();
    		const auto spec_begin = j->second.cbegin();
    		const auto spec_end = j->second.cend();
    		const auto vacancy_iterator = (j->second).find("VA"); // this may point to spec_end
    		/*
    		 * To make sure all mole fractions sum to 1,
    		 * we have to normalize everything using the same
    		 * sublattices. That means even species which don't
    		 * appear in a given sublattice need to have that
    		 * sublattice's coefficient appear in its denominator.
    		 * To accomplish this task, we perform two loops in each sublattice:
    		 * 1) all species (except VA), to add to the denominator
    		 * 2) only species in that sublattice, to add to the numerator
    		 * With this method, all mole fractions will sum properly.
    		 */
    		for (auto k = cond_spec_begin; k != cond_spec_end; ++k) {
    			if (*k == "VA") continue; // vacancies don't contribute to mole fractions
				if (vacancy_iterator != spec_end) {
					phase_comp[*k].second += den * (1 - vacancy_iterator->second);
					global_comp[*k].second += phasefrac * den * (1 - vacancy_iterator->second);
				}
				else {
					phase_comp[*k].second += den;
					global_comp[*k].second += phasefrac * den;
				}
    		}
    		for (auto k = spec_begin; k != spec_end; ++k) {
    			if (k->first == "VA") continue; // vacancies don't contribute to mole fractions
                double num = k->second * stoi_coef;
                phase_comp[k->first].first += num;
                global_comp[k->first].first += phasefrac * num;
    		}
    	}
    	const auto cmp_begin = phase_comp.cbegin();
    	const auto cmp_end = phase_comp.cend();
    	for (auto g = cmp_begin; g != cmp_end; ++g) {
    		temp_buf << g->first << " " << (g->second.first / g->second.second) << "  ";
    	}
    	temp_buf << std::endl << "Constitution:" << std::endl;

    	for (auto j = subl_begin; j != subl_end; ++j) {
    		double stoi_coef = j->first;
    		temp_buf << "Sublattice " << std::distance(subl_begin,j)+1 << ", Number of sites " << stoi_coef << std::endl;
    		const auto spec_begin = j->second.cbegin();
    		const auto spec_end = j->second.cend();
    		for (auto k = spec_begin; k != spec_end; ++k) {
                temp_buf << k->first << " " << k->second << " ";
    		}
    		temp_buf << std::endl;
    	}

    	// if we're at the last phase, don't add an extra newline
    	if (std::distance(i,ph_end) != 1) temp_buf << std::endl;
    }
    const auto glob_begin = global_comp.cbegin();
    const auto glob_end = global_comp.cend();
    stream << "Component\tMoles\tW-Fraction\tActivity\tPotential\tRef.state" << std::endl;
    for (auto h = glob_begin; h != glob_end; ++h) {
    	stream << h->first << " " << (h->second.first / h->second.second) * N << " ???? ???? ???? ????" << std::endl;
    }
    stream << std::endl;

    stream << temp_buf.rdbuf(); // include the temporary buffer with all the phase data

	return stream;
}
