// hessian.cpp -- definition for the Hessian (second derivative matrix) calculation

#include "optimizer.h"

using namespace arma;

void calculate_hessian(
	const Phase_Collection::const_iterator phase_iter,
	const Phase_Collection::const_iterator phase_end,
	const vector_map &var_map,
	const vector &variables,
	matrix &Hessian,
	const evalconditions &conditions
	) {
			// Build a sitefracs object so that we can reference site fractions conveniently
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


	// Calculate the Hessian matrix.
	// Now the Hessian is a sparse matrix. We're only going to calculate the
	//    nonzero terms and assume it was initialized as a zero matrix.
	// Because Hessian is a symmetric_matrix, cross terms only need
	//    to be populated in the upper or lower triangular section. uBLAS will
	//    account for the other term automatically.
	auto sitefrac_begin = var_map.sitefrac_iters.begin();
	int lambda2p_var_index = 0;
	std::cout << "HESSIAN: start phase loop" << std::endl;
	// all phases
	for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
		sublattice_vector::const_iterator subls_start = (mysitefracs[std::distance(sitefrac_begin,i)].second).cbegin();
		sublattice_vector::const_iterator subls_end = (mysitefracs[std::distance(sitefrac_begin,i)].second).cend();
		const Phase_Collection::const_iterator cur_phase_iter = 
			var_map.phasefrac_iters[std::distance(sitefrac_begin,i)].get<2>(); // iterator for the Phase object

		int phasefrac_var_index = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<0>();
		long double fL = variables[phasefrac_var_index]; // phase fraction

		Hessian(phasefrac_var_index, lambda2p_var_index) = 1;
		Hessian(lambda2p_var_index, phasefrac_var_index) = 1;
		std::cout << "HESSIAN: start conditions loop" << std::endl;
		for (auto m = conditions.elements.cbegin(); m != conditions.elements.cend(); ++m) {
			// TODO: support for X(phase,component) constraints; will need to test for that case here
			const int lambda1p_var_index = var_map.lambda1p_iters[std::distance(conditions.elements.cbegin(),m)].first;
			Hessian(phasefrac_var_index, lambda1p_var_index) = mole_fraction(
				(*m),
				cur_phase_iter->second.get_sublattice_iterator(),
				cur_phase_iter->second.get_sublattice_iterator_end(),
				subls_start,
				subls_end
				);
			Hessian(lambda1p_var_index, phasefrac_var_index) = Hessian(phasefrac_var_index, lambda1p_var_index);
		}

		// each sublattice
		auto subl_begin = (*i).begin();
		std::cout << "HESSIAN: start subl loop" << std::endl;
		for (auto j = subl_begin; j != (*i).end(); ++j) {
			int sublindex = std::distance(subl_begin, j);
			// each species
			auto spec_begin = (*j).begin();
			std::cout << "HESSIAN: start species loop" << std::endl;
			for (auto k = spec_begin; k != (*j).end(); ++k) {
				const int sitefrac_var_index = var_map.sitefrac_iters.at(std::distance(sitefrac_begin,i)).at(sublindex).at(k->first).first; 
				double dfLdy = get_Gibbs_deriv(
					subls_start,
					subls_end,
					cur_phase_iter,
					conditions,
					sublindex,
					k->first
					); // init this second derivative with the derivative of G w.r.t y

				std::cout << "HESSIAN: start conditions loop" << std::endl;
				for (auto m = conditions.elements.cbegin(); m != conditions.elements.cend(); ++m) {
					// TODO: support for X(phase,component) constraints; will need to test for that case here
					const int lambda1p_var_index = var_map.lambda1p_iters[std::distance(conditions.elements.cbegin(),m)].first;
					double lambda1p = variables[lambda1p_var_index];
					dfLdy += lambda1p * mole_fraction_deriv(
							(*m),
							k->first,
							sublindex,
							cur_phase_iter->second.get_sublattice_iterator(),
							cur_phase_iter->second.get_sublattice_iterator_end(),
							subls_start,
							subls_end
							);

					double sum_xfrac_derivs = 0;
					// need to iterate over all phases for this summation
					std::cout << "HESSIAN: start inner phase summation loop" << std::endl;
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
					Hessian(sitefrac_var_index, lambda1p_var_index) = fL * sum_xfrac_derivs;
					Hessian(lambda1p_var_index, sitefrac_var_index) = Hessian(sitefrac_var_index, lambda1p_var_index);
				}
				Hessian(sitefrac_var_index, phasefrac_var_index) = dfLdy;
				Hessian(phasefrac_var_index, sitefrac_var_index) = dfLdy;
				const int lambda3p_var_index = 
					var_map.lambda3p_iters[std::distance(sitefrac_begin,i) + std::distance(subl_begin,j) + std::distance(spec_begin,k)].first;
				double stoi_coef = cur_phase_iter->second.subls[sublindex].stoi_coef;
				Hessian(sitefrac_var_index, lambda3p_var_index) = 1;
				Hessian(lambda3p_var_index, sitefrac_var_index) = 1;
				// TODO: expanded second derivative of G for non-ideal interactions
				if (variables[sitefrac_var_index] > 0) Hessian(sitefrac_var_index, sitefrac_var_index) = 
					fL * SI_GAS_CONSTANT * conditions.statevars.at('T') * stoi_coef / variables[sitefrac_var_index];
			}
		}
	}
}