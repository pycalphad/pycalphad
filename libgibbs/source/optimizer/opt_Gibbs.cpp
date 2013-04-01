/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// opt_Gibbs.cpp -- definition for Gibbs energy optimizer

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/optimizer.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include "external/coin/IpTNLP.hpp"

using namespace Ipopt;

/* Constructor. */
GibbsOpt::GibbsOpt(
	const Phase_Collection::const_iterator p_begin, 
	const Phase_Collection::const_iterator p_end, 
	const evalconditions &sysstate) {
	conditions = sysstate;
	phase_iter = p_begin;
	phase_end = p_end;
	int varcount = 0;
	// Build a sitefracs object so that we can calculate the Gibbs energy
	for (auto i = phase_iter; i != phase_end; ++i) {
		sublattice_vector subls_vec;
		for (auto j = i->second.get_sublattice_iterator(); j != i->second.get_sublattice_iterator_end();++j) {
			std::map<std::string,double> subl_map;
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					subl_map[*k] = 1;
					//var_map.sitefrac_iters[std::distance(phase_iter,i)][std::distance(i->second.get_sublattice_iterator(),j)][*k]
				}
			}
			subls_vec.push_back(subl_map);
		}
		mysitefracs.push_back(std::make_pair(i->first,subls_vec));
	}
	// Build the index map
	for (auto i = phase_iter; i != phase_end; ++i) {
		////std::cout << "x[" << varcount << "] = " << i->first << " phasefrac" << std::endl;
		var_map.phasefrac_iters.push_back(boost::make_tuple(varcount,varcount+1,i));
		++varcount;
		for (auto j = i->second.get_sublattice_iterator(); j != i->second.get_sublattice_iterator_end();++j) {
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					// This site matches one of our elements under investigation
					// Add it to the list of sitefracs
					// +1 for a sitefraction
					//std::cout << "x[" << varcount << "] = (" << i->first << "," << std::distance(i->second.get_sublattice_iterator(),j) << "," << *k << ")" << std::endl;
					var_map.sitefrac_iters.resize(std::distance(phase_iter,i)+1);
					var_map.sitefrac_iters[std::distance(phase_iter,i)].resize(std::distance(i->second.get_sublattice_iterator(),j)+1);
					var_map.sitefrac_iters[std::distance(phase_iter,i)][std::distance(i->second.get_sublattice_iterator(),j)][*k] =
						std::make_pair(varcount,i);
					++varcount;
				}
			}
		}
	}
}

GibbsOpt::~GibbsOpt()
{}

bool GibbsOpt::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                         Index& nnz_h_lag, IndexStyleEnum& index_style)
{
  Index sublcount = 0;
  Index phasecount = 0;
  Index sitefraccount = 0;
  Index speccount = (Index) conditions.xfrac.size();
  auto sitefrac_begin = var_map.sitefrac_iters.cbegin();
  for (auto i = var_map.sitefrac_iters.cbegin(); i != var_map.sitefrac_iters.cend(); ++i) {
	  const Phase_Collection::const_iterator cur_phase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();
	  const Sublattice_Collection::const_iterator subls_start = cur_phase->second.get_sublattice_iterator();
	  const Sublattice_Collection::const_iterator subls_end = cur_phase->second.get_sublattice_iterator_end();
	  sublcount += std::distance(subls_start,subls_end);
	  for (auto j = subls_start; j != subls_end;++j) {
		  for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
			  // Check if this species in this sublattice is on our list of elements to investigate
			  if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
				  ++sitefraccount;
			  }
		  }
	  }
	  ++phasecount;
  }

  // number of variables
  n = sitefraccount + phasecount;
   // one phase fraction balance equality constraint, plus all the sublattice fraction balance constraints
   // plus all the mass balance constraints
  //std::cout << "n = " << sitefraccount << " + " << phasecount << std::endl;
  //std::cout << "m = 1 + " << sublcount << " + " << speccount << std::endl;
  m = 1 + sublcount + speccount;

  // nonzeros in the jacobian of the lagrangian
  nnz_jac_g = phasecount + speccount*phasecount + 2 * sitefraccount; // this is potentially an overestimate

  // nonzeros in the hessian of the lagrangian
  //nnz_h_lag = (phasecount * sitefraccount) + phasecount * (1 + speccount) + sitefraccount * (1 + sublcount + speccount);

  index_style = C_STYLE;
  return true;
}

bool GibbsOpt::get_bounds_info(Index n, Number* x_l, Number* x_u,
                            Index m_num, Number* g_l, Number* g_u)
{
	// We initialize the bounds but, once the solver is running,
	// we only reset the bounds to fix inconsistencies.
	// site and phase fractions have a lower bound of 0 and an upper bound of 1
	for (Index i = 0; i < n; ++i) {
		//if (x_u[i] < x_l[i] || x_l[i] == NULL || x_u[i] == NULL) {
			x_l[i] = 0;
			x_u[i] = 1;
		//}
	}

	Index cons_index = 0;
	// Phase fraction balance constraint
	//if (g_u[cons_index] < g_l[cons_index] || g_l[cons_index] == NULL || g_u[cons_index] == NULL) {
		g_l[cons_index] = -1;
		g_u[cons_index] = 0;
	//}
	++cons_index;
	auto sitefrac_begin = var_map.sitefrac_iters.begin();
	for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
		const Phase_Collection::const_iterator cur_phase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();
		for (auto j = cur_phase->second.get_sublattice_iterator(); j != cur_phase->second.get_sublattice_iterator_end();++j) {
			// Site fraction balance constraint
			//if (g_u[cons_index] < g_l[cons_index] || g_l[cons_index] == NULL || g_u[cons_index] == NULL) {
				g_l[cons_index] = -1;
				g_u[cons_index] = 0;
			//}
			++cons_index;
		}
	}

	// Mass balance constraint

	for (auto i = 0; i < conditions.xfrac.size(); ++i) {
		if (g_u[cons_index] < g_l[cons_index] || g_l[cons_index] == NULL || g_u[cons_index] == NULL) {
			g_l[cons_index] = -1;
			g_u[cons_index] = 0;
		}
		++cons_index;
	}

	assert(m_num == cons_index); // TODO: rewrite as exception
	return true;
}

bool GibbsOpt::get_starting_point(Index n, bool init_x, Number* x,
	bool init_z, Number* z_L, Number* z_U,
	Index m, bool init_lambda,
	Number* lambda)
{
	for (Index i = 0; i < n; ++i) {
		x[i] = 0;
	}
	return true;
}

bool GibbsOpt::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
	// return the value of the objective function
	auto sitefrac_begin = var_map.sitefrac_iters.begin();
	double result = 0;
	// all phases
	for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
		const int phaseindex = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<0>();
		const Phase_Collection::const_iterator cur_phase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();
		const double fL = x[phaseindex]; // phase fraction

		sublattice_vector subls_vec;
		for (auto j = cur_phase->second.get_sublattice_iterator(); j != cur_phase->second.get_sublattice_iterator_end();++j) {
			std::map<std::string,double> subl_map;
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					subl_map[*k] = x[var_map.sitefrac_iters[std::distance(sitefrac_begin,i)][std::distance(cur_phase->second.get_sublattice_iterator(),j)][*k].first];
				}
			}
			subls_vec.push_back(subl_map);
		}
		sublattice_vector::const_iterator subls_start = subls_vec.cbegin();
		sublattice_vector::const_iterator subls_end = subls_vec.cend();


		double temp = get_Gibbs(subls_start, subls_end, cur_phase, conditions);
		////std::cout << "eval_f: result = " << fL << " * " << temp << " = " << fL * temp << std::endl;
		result += fL * temp;
	}
	obj_value = result;
	return true;
}

bool GibbsOpt::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{
  // return the gradient of the objective function grad_{x} f(x)

	// calculate dF/dy(l,s,j)
	auto sitefrac_begin = var_map.sitefrac_iters.begin();
	int varcheck = 0;
	// all phases
	for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
		int phaseindex = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<0>();
		const Phase_Collection::const_iterator cur_phase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();
		double fL = x[phaseindex]; // phase fraction

		sublattice_vector subls_vec;
		for (auto j = cur_phase->second.get_sublattice_iterator(); j != cur_phase->second.get_sublattice_iterator_end();++j) {
			std::map<std::string,double> subl_map;
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					subl_map[*k] = x[var_map.sitefrac_iters[std::distance(sitefrac_begin,i)][std::distance(cur_phase->second.get_sublattice_iterator(),j)][*k].first];
				}
			}
			subls_vec.push_back(subl_map);
		}
		sublattice_vector::const_iterator subls_start = subls_vec.cbegin();
		sublattice_vector::const_iterator subls_end = subls_vec.cend();

		// calculate dF/dfL
		double Gibbs = get_Gibbs(subls_start, subls_end, cur_phase,conditions);
		//std::cout << "grad_f[" << phaseindex << "] = " << Gibbs << std::endl;
		grad_f[phaseindex] = Gibbs; ++varcheck;

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
				double dGdy = get_Gibbs_deriv(
					subls_start,
					subls_end,
					phase_iter,
					conditions,
					sublindex,
					k->first
					);
				// k->second.first = index
				grad_f[k->second.first] = fL * dGdy; ++varcheck;
				//std::cout << "grad_f[" << k->second.first << "] = " << fL << " * " << dGdy << " = " << fL * dGdy << std::endl;
			}
		}
	}
	assert (varcheck == n);
	// calculate dF/dfL
	/*for (auto i = var_map.phasefrac_iters.begin(); i != var_map.phasefrac_iters.end(); ++i) {
		sublattice_vector::const_iterator subls_start = (mysitefracs[std::distance(var_map.phasefrac_iters.begin(),i)].second).cbegin();
		sublattice_vector::const_iterator subls_end = (mysitefracs[std::distance(var_map.phasefrac_iters.begin(),i)].second).cend();
		double Gibbs = get_Gibbs(subls_start, subls_end, (*i).get<2>(),conditions);
		grad_f[(*i).get<0>()] = Gibbs;
	}*/

  return true;
}

bool GibbsOpt::eval_g(Index n, const Number* x, bool new_x, Index m_num, Number* g)
{
	//std::cout << "entering eval_g" << std::endl;
  // return the value of the constraints: g(x)
	double sum_phase_fracs = 0;
	Index cons_index = 1;
	sitefracs thesitefracs;
	auto sitefrac_begin = var_map.sitefrac_iters.begin();
	for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
		const Index phaseindex = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<0>();
		const Phase_Collection::const_iterator cur_phase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();
		const double fL = x[phaseindex]; // phase fraction
		//std::cout << "sum_phase_fracs += (x[" << phaseindex << "] = " << fL << ")" << std::endl;
		sum_phase_fracs += fL;

		sublattice_vector subls_vec;
		for (auto j = cur_phase->second.get_sublattice_iterator(); j != cur_phase->second.get_sublattice_iterator_end();++j) {
			double sum_site_fracs = 0;
			std::map<std::string,double> subl_map;
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					int sitefracindex = var_map.sitefrac_iters[std::distance(sitefrac_begin,i)][std::distance(cur_phase->second.get_sublattice_iterator(),j)][*k].first;
					subl_map[*k] = x[sitefracindex];
					////std::cout << "Sublattice " << std::distance(cur_phase->second.get_sublattice_iterator(),j) << std::endl;
					////std::cout << "subl_map[" << *k << "] = x[" << sitefracindex << "] = " << x[sitefracindex] << std::endl;
					sum_site_fracs  += subl_map[*k];
				}
			}
			// Site fraction balance constraint
			//std::cout << "g[" << cons_index << "] = " << sum_site_fracs << " - " << "1 = " << sum_site_fracs - 1 << std::endl;
			g[cons_index] = sum_site_fracs - 1;
			++cons_index;
			subls_vec.push_back(subl_map);
		}
		thesitefracs.push_back(std::make_pair(cur_phase->first,subls_vec));

	}
	// Phase fraction balance constraint
	//std::cout << "g[0] = " << sum_phase_fracs << " - 1 = " << sum_phase_fracs - 1 << std::endl;
	g[0] = sum_phase_fracs - 1;

	// Mass balance constraint
	for (auto m = conditions.xfrac.cbegin(); m != conditions.xfrac.cend(); ++m) {
		double sumterm = 0;
		for (auto i = var_map.phasefrac_iters.begin(); i != var_map.phasefrac_iters.end(); ++i) {
			const Phase_Collection::const_iterator myphase = (*i).get<2>();
			sublattice_vector::const_iterator subls_start = (thesitefracs[std::distance(var_map.phasefrac_iters.begin(),i)].second).cbegin();
			sublattice_vector::const_iterator subls_end = (thesitefracs[std::distance(var_map.phasefrac_iters.begin(),i)].second).cend();
			double molefrac = mole_fraction(
				(*m).first,
				(*myphase).second.get_sublattice_iterator(),
				(*myphase).second.get_sublattice_iterator_end(),
				subls_start,
				subls_end
				);
			// sumterm += fL * molefrac
			////std::cout << "sumterm += (" << x[(*i).get<0>()] << " * " << molefrac << " = " << x[(*i).get<0>()] * molefrac << ")" << std::endl;
			sumterm += x[(*i).get<0>()] * molefrac;
		}
		// Mass balance
		//std::cout << "g[" << cons_index << "] = " << sumterm << " - " << m->second << " = " << (sumterm - m->second) << std::endl;
		g[cons_index] = sumterm - m->second;
		++cons_index;
	}
	assert(cons_index == m_num);
	//std::cout << "exiting eval_g" << std::endl;
  return true;
}

bool GibbsOpt::eval_jac_g(Index n, const Number* x, bool new_x,
	Index m_num, Index nele_jac, Index* iRow, Index *jCol,
	Number* values)
{
	if (values == NULL) {
		//std::cout << "entering eval_jac_g values == NULL" << std::endl;
		Index cons_index = 0;
		Index jac_index = 0;
		auto sitefrac_begin = var_map.sitefrac_iters.begin();
		sitefracs thesitefracs;
		for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
			const Index phaseindex = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<0>();
			const Phase_Collection::const_iterator cur_phase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();

			// Phase fraction balance constraint
			//std::cout << "iRow[" << jac_index << "] = 0; jCol[" << jac_index << "] = " << phaseindex << std::endl;
			iRow[jac_index] = 0;
			jCol[jac_index] = phaseindex;
			//values[jac_index] = 1;
			++jac_index;

			double sum_site_fracs = 0;
			sublattice_vector subls_vec;
			for (auto j = cur_phase->second.get_sublattice_iterator(); j != cur_phase->second.get_sublattice_iterator_end();++j) {
				std::map<std::string,double> subl_map;
				++cons_index;
				for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
					// Check if this species in this sublattice is on our list of elements to investigate
					if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
						Index sitefracindex;
						sitefracindex = var_map.sitefrac_iters[std::distance(sitefrac_begin,i)][std::distance(cur_phase->second.get_sublattice_iterator(),j)][*k].first;
						////std::cout << "sitefracindex = " << sitefracindex << std::endl;
						//subl_map[*k] = x[sitefracindex];
						//sum_site_fracs  += subl_map[*k];
						// Site fraction balance constraint
						//std::cout << "iRow[" << jac_index << "] = " << cons_index << "; jCol[" << jac_index << "] = " << sitefracindex << std::endl;
						iRow[jac_index] = cons_index; 
						jCol[jac_index] = sitefracindex;
						//values[jac_index] = 1;
						++jac_index;
					}
				}
				subls_vec.push_back(subl_map);
			}
			thesitefracs.push_back(std::make_pair(cur_phase->first,subls_vec));
		}
		++cons_index;

		// Mass balance constraint
		for (auto m = conditions.xfrac.cbegin(); m != conditions.xfrac.cend(); ++m) {
			for (auto i = var_map.phasefrac_iters.begin(); i != var_map.phasefrac_iters.end(); ++i) {
				const Phase_Collection::const_iterator myphase = (*i).get<2>();
				sublattice_vector::const_iterator subls_start = (mysitefracs[std::distance(var_map.phasefrac_iters.begin(),i)].second).cbegin();
				sublattice_vector::const_iterator subls_end = (mysitefracs[std::distance(var_map.phasefrac_iters.begin(),i)].second).cend();
				const Index phaseindex = (*i).get<0>();
				//const double fL = x[phaseindex];
				/*double molefrac = mole_fraction(
					(*m),
					(*myphase).second.get_sublattice_iterator(),
					(*myphase).second.get_sublattice_iterator_end(),
					subls_start,
					subls_end
					);*/
				// Mass balance constraint, w.r.t phase fraction
				//std::cout << "iRow[" << jac_index << "] = " << cons_index << "; jCol[" << jac_index << "] = " << phaseindex << std::endl;
				iRow[jac_index] = cons_index;
				jCol[jac_index] = phaseindex;
				//values[jac_index] = molefrac;
				++jac_index;
			}
			auto sitefrac_begin = var_map.sitefrac_iters.begin();
			for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
				const Index phaseindex = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<0>();
				const Phase_Collection::const_iterator cur_phase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();
				//const double fL = x[phaseindex]; // phase fraction

				sublattice_vector subls_vec;
				for (auto j = cur_phase->second.get_sublattice_iterator(); j != cur_phase->second.get_sublattice_iterator_end();++j) {
					std::map<std::string,double> subl_map;
					int sublindex = std::distance(cur_phase->second.get_sublattice_iterator(),j);
					for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
						// Check if this species in this sublattice is on our list of elements to investigate
						if ((*k) == (*m).first) {
							Index sitefracindex = var_map.sitefrac_iters[std::distance(sitefrac_begin,i)][sublindex][*k].first;
							//subl_map[*k] = x[sitefracindex];
							//std::cout << "iRow[" << jac_index << "] = " << cons_index << "; jCol[" << jac_index << "] = " << sitefracindex << std::endl;
							iRow[jac_index] = cons_index;
							jCol[jac_index] = sitefracindex;
							// values[jac_index] = fL * molefrac_deriv;
							++jac_index;
						}
					}
					subls_vec.push_back(subl_map);
				}
			}
			// Mass balance
			++cons_index;
		}
		assert (cons_index == m_num);
		//std::cout << "assertion succeeded" << std::endl;
	}
	else {
		//std::cout << "entering eval_jac_g with values" << std::endl;
		Index cons_index = 0;
		Index jac_index = 0;
		auto sitefrac_begin = var_map.sitefrac_iters.begin();
		sitefracs thesitefracs;
		for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
			const Index phaseindex = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<0>();
			const Phase_Collection::const_iterator cur_phase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();
			double fL = x[phaseindex]; // phase fraction

			// Phase fraction balance constraint
			//iRow[jac_index] = 0;
			//jCol[jac_index] = phaseindex;
			//std::cout << "jac_g values[" << jac_index << "] = 1" << std::endl;
			values[jac_index] = 1;
			++jac_index;

			sublattice_vector subls_vec;
			for (auto j = cur_phase->second.get_sublattice_iterator(); j != cur_phase->second.get_sublattice_iterator_end();++j) {
				std::map<std::string,double> subl_map;
				++cons_index;
				for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
					// Check if this species in this sublattice is on our list of elements to investigate
					Index sitefracindex;
					if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
						sitefracindex = var_map.sitefrac_iters[std::distance(sitefrac_begin,i)][std::distance(cur_phase->second.get_sublattice_iterator(),j)][*k].first;
						subl_map[*k] = x[sitefracindex];
						////std::cout << "subl_map[" << *k << "] = x[" << sitefracindex << "] = " << x[sitefracindex] << std::endl;
						// Site fraction balance constraint
						//iRow[jac_index] = cons_index; 
						//jCol[jac_index] = sitefracindex;
						//std::cout << "jac_g values[" << jac_index << "] = 1" << std::endl;
						values[jac_index] = 1;
						++jac_index;
					}
				}
				subls_vec.push_back(subl_map);
			}
			thesitefracs.push_back(std::make_pair(cur_phase->first,subls_vec));
		}
		++cons_index;

		// Mass balance constraint
		for (auto m = conditions.xfrac.cbegin(); m != conditions.xfrac.cend(); ++m) {
			for (auto i = var_map.phasefrac_iters.begin(); i != var_map.phasefrac_iters.end(); ++i) {
				const Phase_Collection::const_iterator myphase = (*i).get<2>();
				sublattice_vector::const_iterator subls_start = (thesitefracs[std::distance(var_map.phasefrac_iters.begin(),i)].second).cbegin();
				sublattice_vector::const_iterator subls_end = (thesitefracs[std::distance(var_map.phasefrac_iters.begin(),i)].second).cend();
				const Index phaseindex = (*i).get<0>();
				double fL = x[phaseindex];
				double molefrac = mole_fraction(
					m->first,
					(*myphase).second.get_sublattice_iterator(),
					(*myphase).second.get_sublattice_iterator_end(),
					subls_start,
					subls_end
					);
				// Mass balance constraint, w.r.t phase fraction
				//iRow[jac_index] = cons_index;
				//jCol[jac_index] = phaseindex;
				//std::cout << "jac_g values[" << jac_index << "] = " << molefrac << std::endl;
				values[jac_index] = molefrac;
				++jac_index;
			}
			auto sitefrac_begin = var_map.sitefrac_iters.begin();
			for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
				const Index phaseindex = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<0>();
				const Phase_Collection::const_iterator cur_phase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();
				//const Phase_Collection::const_iterator myphase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();
				sublattice_vector::const_iterator subls_start = (thesitefracs[std::distance(sitefrac_begin,i)].second).cbegin();
				sublattice_vector::const_iterator subls_end = (thesitefracs[std::distance(sitefrac_begin,i)].second).cend();
				double fL = x[phaseindex]; // phase fraction

				sublattice_vector subls_vec;
				for (auto j = cur_phase->second.get_sublattice_iterator(); j != cur_phase->second.get_sublattice_iterator_end();++j) {
					std::map<std::string,double> subl_map;
					int sublindex = std::distance(cur_phase->second.get_sublattice_iterator(),j);
					for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
						// Check if this species in this sublattice is on our list of elements to investigate
						if ((*k) == (*m).first) {
							Index sitefracindex = var_map.sitefrac_iters[std::distance(sitefrac_begin,i)][sublindex][*k].first;
							subl_map[*k] = x[sitefracindex];
							//iRow[jac_index] = cons_index;
							//jCol[jac_index] = sitefracindex;
							double molefrac_deriv = mole_fraction_deriv(
								(*m).first,
								(*k),
								sublindex,
								cur_phase->second.get_sublattice_iterator(),
								cur_phase->second.get_sublattice_iterator_end(),
								subls_start,
								subls_end
								);
							//std::cout << "jac_g values[" << jac_index << "] = " << fL*molefrac_deriv << std::endl;
							values[jac_index] = fL * molefrac_deriv;
							++jac_index;
						}
					}
				}
			}
			// Mass balance
			++cons_index;
			//std::cout << "eval_jac_g: cons_index is now " << cons_index << std::endl;
		}
		//std::cout << "complete jac_index: " << jac_index << std::endl;
		assert(cons_index == m_num);
	}
	//std::cout << "exiting eval_jac_g" << std::endl;
	return true;
	}

bool GibbsOpt::eval_h(Index n, const Number* x, bool new_x,
                   Number obj_factor, Index m, const Number* lambda,
                   bool new_lambda, Index nele_hess, Index* iRow,
                   Index* jCol, Number* values)
{
	// TODO: fix
  for (Index i = 0; i < n; ++i) {
	  values[i] = 0;
  }

  return false;
}

void GibbsOpt::finalize_solution(SolverReturn status,
                              Index n, const Number* x, const Number* z_L, const Number* z_U,
                              Index m_num, const Number* g, const Number* lambda,
                              Number obj_value,
			      const IpoptData* ip_data,
			      IpoptCalculatedQuantities* ip_cq)
{
	auto sitefrac_begin = var_map.sitefrac_iters.begin();
	sitefracs thesitefracs;
	for (auto i = sitefrac_begin; i != var_map.sitefrac_iters.end(); ++i) {
		const Index phaseindex = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<0>();
		const Phase_Collection::const_iterator cur_phase = var_map.phasefrac_iters.at(std::distance(sitefrac_begin,i)).get<2>();
		const double fL = x[phaseindex]; // phase fraction

		constitution subls_vec;
		for (auto j = cur_phase->second.get_sublattice_iterator(); j != cur_phase->second.get_sublattice_iterator_end();++j) {
			std::unordered_map<std::string,double> subl_map;
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				Index sitefracindex;
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					sitefracindex = var_map.sitefrac_iters[std::distance(sitefrac_begin,i)][std::distance(cur_phase->second.get_sublattice_iterator(),j)][*k].first;
					subl_map[*k] = x[sitefracindex];
					//std::cout << "y(" << cur_phase->first << "," << *k << ") = " << x[sitefracindex] << std::endl;
				}
			}
			subls_vec.push_back(std::make_pair((*j).stoi_coef,subl_map));
		}
		//thesitefracs.push_back(std::make_pair(cur_phase->first,subls_vec));
		ph_map[cur_phase->first] = std::make_pair(fL,subls_vec);
	}
	/*for (auto m = conditions.elements.cbegin(); m != conditions.elements.cend(); ++m) {
		for (auto i = var_map.phasefrac_iters.begin(); i != var_map.phasefrac_iters.end(); ++i) {
			const Phase_Collection::const_iterator myphase = (*i).get<2>();
			sublattice_vector::const_iterator subls_start = (thesitefracs[std::distance(var_map.phasefrac_iters.begin(),i)].second).cbegin();
			sublattice_vector::const_iterator subls_end = (thesitefracs[std::distance(var_map.phasefrac_iters.begin(),i)].second).cend();
			const Index phaseindex = (*i).get<0>();
			const double fL = x[phaseindex];
			double molefrac = mole_fraction(
				(*m),
				(*myphase).second.get_sublattice_iterator(),
				(*myphase).second.get_sublattice_iterator_end(),
				subls_start,
				subls_end
				);
			//std::cout << "x(" << myphase->first << "," << *m << ") = " << molefrac << " ; X = " << fL*molefrac << std::endl;
		}
	}*/
	for (Index i = 0; i < m_num; ++i) {
		//std::cout << "g[" << i << "] = " << g[i] << std::endl;
	}
}
