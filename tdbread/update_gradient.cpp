// update_gradient.cpp -- definition for the Gibbs energy optimizer gradient update procedure

#include "optimizer.h"

using namespace std;


void update_gradient(
	const Phase_Collection::const_iterator phase_iter,
	const Phase_Collection::const_iterator phase_end,
	const vector_map &var_map,
	vector &variables,
	vector &gradient,
	const evalconditions &conditions
	) {
			// Build a sitefracs object so that we can calculate the Gibbs energy
	sitefracs mysitefracs;
	for (auto i = phase_iter; i != phase_end; ++i) {
		sublattice_vector subls_vec;
		for (auto j = i->second.get_sublattice_iterator(); j != i->second.get_sublattice_iterator_end();++j) {
			std::map<std::string,double> subl_map;
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					subl_map[*k] = 
						variables[
							var_map.sitefrac_iters[std::distance(phase_iter,i)][std::distance(i->second.get_sublattice_iterator(),j)].at(*k).first
						];
				}
			}
			subls_vec.push_back(subl_map);
		}
		mysitefracs.push_back(std::make_pair(i->first,subls_vec));
	}

	// calculate dF/dy(l,s,j)
	auto sitefrac_begin = var_map.sitefrac_iters.begin();

	// all phases
	for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
		sublattice_vector::const_iterator subls_start = (mysitefracs[std::distance(sitefrac_begin,i)].second).cbegin();
		sublattice_vector::const_iterator subls_end = (mysitefracs[std::distance(sitefrac_begin,i)].second).cend();

		int phaseindex = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<0>();
		long double fL = variables[phaseindex]; // phase fraction
		// update the inequality constraint parameters for fL
		// they are located at phaseindex+1 and phaseindex+2, respectively, for the phasefrac
		// first, determine if either phase fraction constraint is active
		if (fL <= 0) {
			// fL >= 0 constraint is active
			gradient[phaseindex+1] = fL;
			// the fL <= 1 constraint must be inactive; zero out the Lagrange multiplier
			variables[phaseindex+2] = 0;
		}
		else if (fL >= 1) {
			// the fL >= 1 constraint is active
			gradient[phaseindex+2] = 1 - fL;
			// the fL <= 0 constraint must be inactive; zero out the Lagrange multiplier
			variables[phaseindex+1] = 0;
			gradient[phaseindex+1] = 0;
		}
		else {
			// neither constraint is active; zero out Lagrange multipliers
			variables[phaseindex+1] = 0;
			gradient[phaseindex+1] = 0;
			variables[phaseindex+2] = 0;
			gradient[phaseindex+2] = 0;
		}

		// each sublattice
		auto subl_begin = (*i).begin();
		for (auto j = subl_begin; j != (*i).end(); ++j) {
			int sublindex = std::distance(subl_begin, j);
			// each species
			auto spec_begin = (*j).begin();
			for (auto k = spec_begin; k != (*j).end(); ++k) {
				// k->second.second = phase_iter
				// k->second.second->first = name of the phase
				const Phase_Collection::const_iterator phase_iter = k->second.second;
				std::cout << k->second.second->first << "[" << std::distance(sitefrac_begin,i) << "][" << sublindex << "(" << k->first << ")]" << std::endl;
				double dGdy = get_Gibbs_deriv(
					subls_start,
					subls_end,
					phase_iter,
					conditions,
					sublindex,
					k->first
					);
				std::cout << "\tfL: " << fL << std::endl;
				std::cout << "\tdGdy: " << dGdy << std::endl;

				double element_contrib = 0;
				for (auto m = conditions.elements.cbegin(); m != conditions.elements.cend(); ++m) {
					// TODO: support for X(phase,component) constraints; will need to test for that case here
					double lambda1p = variables[var_map.lambda1p_iters[std::distance(conditions.elements.cbegin(),m)].first];
					double sum_xfrac_derivs = 0;
					// need to iterate over all phases for this summation
					for (auto n = sitefrac_begin; n != var_map.sitefrac_iters.end(); ++n) {
						const Phase_Collection::const_iterator mini_phase_iter = var_map.phasefrac_iters[std::distance(sitefrac_begin,n)].get<2>();
						sublattice_vector::const_iterator mini_subls_start = (mysitefracs[std::distance(sitefrac_begin,n)].second).cbegin();
						sublattice_vector::const_iterator mini_subls_end = (mysitefracs[std::distance(sitefrac_begin,n)].second).cend();
						sum_xfrac_derivs += mole_fraction_deriv(
							(*m),
							k->first,
							sublindex,
							mini_phase_iter->second.get_sublattice_iterator(),
							mini_phase_iter->second.get_sublattice_iterator_end(),
							mini_subls_start,
							mini_subls_end
							);
					}
					element_contrib += lambda1p * fL * sum_xfrac_derivs;
				}
				// update the inequality constraint parameters for y(l,s,j) while we're here
				long double ysj = variables[k->second.first];
				// they are located at index+1 and index+2, respectively
				// first, determine if either site fraction constraint is active
				if (ysj <= 0) {
					// ysj >= 0 constraint is active
					gradient[k->second.first+1] = ysj;
					// the ysj <= 1 constraint must be inactive; zero out the Lagrange multiplier
					variables[k->second.first+2] = 0;
				}
				else if (ysj >= 1) {
					// the ysj >= 1 constraint is active
					gradient[k->second.first+2] = 1 - ysj;
					// the ysj <= 0 constraint must be inactive; zero out the Lagrange multiplier
					variables[k->second.first+1] = 0;
					gradient[k->second.first+1] = 0;
				}
				else {
					// neither constraint is active; zero out Lagrange multipliers
					variables[k->second.first+1] = 0;
					gradient[k->second.first+1] = 0;
					variables[k->second.first+2] = 0;
					gradient[k->second.first+2] = 0;
				}

				double lambda3p = variables[
					var_map.lambda3p_iters[std::distance(sitefrac_begin,i) + std::distance(subl_begin,j) + std::distance(spec_begin,k)].first
					];
				double eta1p = variables[k->second.first+1];
				double eta2p = variables[k->second.first+2];
				// gradient = fL * dG/dy(l,s,j) + element_contrib + lambda3p + eta1p - eta2p
				// k->second.first = index
				gradient[k->second.first] = fL * dGdy + element_contrib + lambda3p + eta1p - eta2p;
				std::cout << "dF/dy(" << std::distance(sitefrac_begin,i) << ")(" << std::distance(subl_begin,j) << ")(" << std::distance(spec_begin,k) << ") = " << fL << " * " << dGdy << " + " << element_contrib << " + " << lambda3p << " + " << eta1p << " - " << eta2p << std::endl;
			}
		}
	}
	std::cout << "dF/dy calculated" << std::endl;

	// calculate dF/dfL
	for (auto i = var_map.phasefrac_iters.begin(); i != var_map.phasefrac_iters.end(); ++i) {
		sublattice_vector::const_iterator subls_start = (mysitefracs[std::distance(var_map.phasefrac_iters.begin(),i)].second).cbegin();
		sublattice_vector::const_iterator subls_end = (mysitefracs[std::distance(var_map.phasefrac_iters.begin(),i)].second).cend();
		double sumterm = 0;
		const Phase_Collection::const_iterator myphase = (*i).get<2>();
		for (auto m = conditions.elements.cbegin(); m != conditions.elements.cend(); ++m) {
			// sumterm += lambda1p * X(phase,component)
			// TODO: support for X(phase,component) constraints; will need to test for that case here
			double l1p = variables[var_map.lambda1p_iters[std::distance(conditions.elements.cbegin(),m)].first];
			double molefrac = mole_fraction(
				(*m),
				(*myphase).second.get_sublattice_iterator(),
				(*myphase).second.get_sublattice_iterator_end(),
				subls_start,
				subls_end
				);
			std::cout << "dF/dfL sumterm += " << l1p << " * " << molefrac << std::endl;
			sumterm += l1p * molefrac;
		}
		double eta1p = variables[(*i).get<0>() + 1]; // fL constraint parameter for fL >=0
		double eta2p = variables[(*i).get<0>() + 2]; // fL constraint parameter for fL <=1
		double Gibbs = get_Gibbs(subls_start, subls_end, (*i).get<2>(),conditions);
			std::cout << "gradient(" << (*i).get<0>() << ") = " << Gibbs << " + " << sumterm << " + " << variables(0) << " + " << eta1p << " - " << eta2p << std::endl;
			gradient[(*i).get<0>()] = 
				 Gibbs + sumterm + variables(0) + eta1p - eta2p;
			std::cout << "dF/dfL(" << std::distance(var_map.phasefrac_iters.begin(),i) << ") = " << gradient[(*i).get<0>()] << std::endl;
	}
	std::cout << "dF/dfL calculated" << std::endl;

	// calculate dF/dlambda1p(m)
	for (auto m = conditions.elements.cbegin(); m != conditions.elements.cend(); ++m) {
		if ((*m) == "VA") continue;
		// TODO: support for X(phase,component) constraints; will need to test for that case here
		double sum_xfrac = 0;
		// need to iterate over all phases for this summation
		for (auto n = sitefrac_begin; n != var_map.sitefrac_iters.end(); ++n) {
			const Phase_Collection::const_iterator mini_phase_iter = var_map.phasefrac_iters[std::distance(sitefrac_begin,n)].get<2>();
			double fL = variables[var_map.phasefrac_iters[std::distance(sitefrac_begin,n)].get<0>()];
			sublattice_vector::const_iterator mini_subls_start = (mysitefracs[std::distance(sitefrac_begin,n)].second).cbegin();
			sublattice_vector::const_iterator mini_subls_end = (mysitefracs[std::distance(sitefrac_begin,n)].second).cend();
			sum_xfrac += fL*mole_fraction(
				(*m),
				mini_phase_iter->second.get_sublattice_iterator(),
				mini_phase_iter->second.get_sublattice_iterator_end(),
				mini_subls_start,
				mini_subls_end
				);
		}
		double global_x_frac = conditions.xfrac.at(*m); // mole fraction for the component over the entire system
		gradient[var_map.lambda1p_iters[std::distance(conditions.elements.cbegin(),m)].first] = sum_xfrac - global_x_frac;

		std::cout << "dF/dlambda1p(" << *m << ") = " << (sum_xfrac - global_x_frac) << std::endl;
	}
	std::cout << "dF/dlambda1p calculated" << std::endl;

	// calculate dF/dlambda2p
	double lambda2p_result = 0;
	for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
		std::cout << "lambda2p: index " << std::distance(sitefrac_begin,i) << std::endl;
		std::cout << "lambda2p: += " << variables[var_map.phasefrac_iters[std::distance(sitefrac_begin,i)].get<0>()] << std::endl;
		lambda2p_result += variables[var_map.phasefrac_iters[std::distance(sitefrac_begin,i)].get<0>()];
	}
	lambda2p_result = lambda2p_result - 1;
	gradient[0] = lambda2p_result; // index 0 is defined as lambda2p
	std::cout << "dF/dlambda2p = " << lambda2p_result << std::endl;
	std::cout << "dF/dlambda2p calculated" << std::endl;

	// calculate dF/dlambda3p
	// this is the site fraction balance
	// as an equality constraint, it's always active
	// we need to iterate over all the sublattices in all phases
	int totalsubls = 0;
	for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
		sublattice_vector::const_iterator subls_start = (mysitefracs[std::distance(sitefrac_begin,i)].second).cbegin();
		sublattice_vector::const_iterator subls_end = (mysitefracs[std::distance(sitefrac_begin,i)].second).cend();
		for (auto s = subls_start; s != subls_end; ++s) {
			double sum_site_frac = 0; // this is the sum of site fractions in this phase
			for (auto k = (*s).cbegin(); k != (*s).cend(); ++k) { sum_site_frac += k->second; } // sum site fractions
			gradient[var_map.lambda3p_iters[totalsubls].first] = sum_site_frac - 1;
			std::cout << "dF/dlambda3p(" << std::distance(sitefrac_begin,i) << ")(" << std::distance(subls_start,s) << ") [" << totalsubls << "] = " << (sum_site_frac - 1) << std::endl;
			++totalsubls;
		}
	}
	std::cout << "dF/dlambda3p calculated" << std::endl;
}