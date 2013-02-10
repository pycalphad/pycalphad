// opt_init_vars.cpp -- definition of initialize_variables() for the Gibbs energy optimizer

#include "optimizer.h"
#include <vector>

using namespace std;

vector initialize_variables(
	const Phase_Collection::const_iterator phase_iter,
	const Phase_Collection::const_iterator phase_end,
	const evalconditions &conditions,
	vector_map &retmap
	) {
		vector_map mymap;
		vector retvec(1);
		for (auto i = retvec.begin(); i != retvec.end(); ++i) {
			(*i) = double(0);
		}
		int resultsize = 1; 
		retvec(0) = 0; // lambda'': site fraction balance parameter
		std::cout << "setting lambda2p variables(0) = " << retvec(0) << std::endl;

	// We need to figure out how big our results vector needs to be
	//int phasecount = std::distance(phase_end,phase_iter); // phases in system

	// variables.size() = phasecount + totalsublcount + sitefraccount + indy_components + constraintcount + 1;
	// phasecount is number of fL terms
	// totalsublcount is accumulated sublattice count
	// sitefraccount is total number of site fractions in all sublattices, in all active phases
	// indy_components is number of element mass balances
	// constraintcount is number of constrained variables
	// Plus 1 phase fraction balance

	// Build the index map
	for (auto i = phase_iter; i != phase_end; ++i) {
		retvec.resize(retvec.size()+1);
		std::cout << "setting phasefrac variables(" << resultsize << ") = " << double(1) / double(std::distance(phase_iter,phase_end)) << std::endl;
		retvec(resultsize) = double(1) / double(std::distance(phase_iter,phase_end)); // set phasefrac to 1/phasecount
		mymap.phasefrac_iters.push_back(boost::make_tuple(resultsize,resultsize+1,i));
		// inequality constraint parameters for phasefrac
		++resultsize;
		retvec.resize(retvec.size()+1);
		retvec(resultsize) = 0;
		std::cout << "setting eta1p variables(" << resultsize << ") = " << retvec(resultsize) << std::endl;
		mymap.eta1p_iters.push_back(std::make_pair(resultsize,resultsize+1));
		++resultsize;
		retvec.resize(retvec.size()+1);
		retvec(resultsize) = 0;
		std::cout << "setting eta2p variables(" << resultsize << ") = " << retvec(resultsize) << std::endl;
		mymap.eta2p_iters.push_back(std::make_pair(resultsize,resultsize+1));
		++resultsize;
		for (auto j = i->second.get_sublattice_iterator(); j != i->second.get_sublattice_iterator_end();++j) {
			retvec.resize(retvec.size()+1);
			retvec(resultsize) = 1;
			mymap.lambda3p_iters.push_back(std::make_pair(resultsize,resultsize+1)); // +1 for an accumulated sublattice
			std::cout << "setting lambda3p variables(" << resultsize << ") = " << retvec(resultsize) << std::endl;
			++resultsize;
			//auto temp_size = resultsize;
			int speccount = 0;
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					// This site matches one of our elements under investigation
					// Add it to the list of sitefracs
					// +1 for a sitefraction
					retvec.resize(retvec.size()+1);
					++speccount;
					for (auto n = 0; n < speccount; ++n) {
						// adjust the site fractions to start as 1 / #species_in_sublattice
						std::cout << "setting sitefrac variables(" << resultsize - 3*n << ") = " << (double(1) / double(speccount)) << std::endl;
						retvec(resultsize - 3*n) = double(1) / double(speccount);
					}
					mymap.sitefrac_iters.resize(std::distance(phase_iter,i)+1);
					mymap.sitefrac_iters[std::distance(phase_iter,i)].resize(std::distance(i->second.get_sublattice_iterator(),j)+1);
					mymap.sitefrac_iters[std::distance(phase_iter,i)][std::distance(i->second.get_sublattice_iterator(),j)][*k] =
						std::make_pair(resultsize,i);
					// inequality constraint parameters for sitefrac
					++resultsize;
					retvec.resize(retvec.size()+1);
					retvec(resultsize) = 0;
					std::cout << "setting eta1p variables(" << resultsize << ") = " << retvec(resultsize) << std::endl;
					mymap.eta1p_iters.push_back(std::make_pair(resultsize,resultsize+1));
					++resultsize;
					retvec.resize(retvec.size()+1);
					retvec(resultsize) = 0;
					std::cout << "setting eta2p variables(" << resultsize << ") = " << retvec(resultsize) << std::endl;
					mymap.eta2p_iters.push_back(std::make_pair(resultsize,resultsize+1));
					++resultsize;
					//std::cout << i->first << " pair produced at [" << std::distance(phase_iter,i) << "][" << std::distance(i->second.get_sublattice_iterator(),j) << "], species " << *k << std::endl;
				}
			}
			//resultsize = temp_size;
		}
	}
	for (auto i = conditions.elements.cbegin(); i != conditions.elements.cend(); ++i) {
		// lambda'(j): element mass balance constraint parameter
		// eta', eta'': inequality constraint parameters
		retvec.resize(retvec.size()+1);
		retvec(resultsize) = 0;
		std::cout << "setting lambda1p variables(" << resultsize << ") = " << retvec(resultsize) << std::endl;
		mymap.lambda1p_iters.push_back(std::make_pair(resultsize,resultsize+1));
		++resultsize;
		/*retvec.resize(retvec.size()+1);
		retvec(resultsize) = 1;
		std::cout << "setting eta1p variables(" << resultsize << ") = " << retvec(resultsize) << std::endl;
		mymap.eta1p_iters.push_back(std::make_pair(resultsize,resultsize+1));
		++resultsize;
		retvec.resize(retvec.size()+1);
		retvec(resultsize) = 1;
		std::cout << "setting eta2p variables(" << resultsize << ") = " << retvec(resultsize) << std::endl;
		mymap.eta2p_iters.push_back(std::make_pair(resultsize,resultsize+1));
		++resultsize;*/
		// +3 for 1 element mass balance constraint plus 2 inequalities (0 <= x <= 1)
	}

	retmap = mymap;
	return retvec;
}