// optimizer.h -- header file for Gibbs energy optimizer
#ifndef INCLUDED_OPTIMIZER
#define INCLUDED_OPTIMIZER

#include "database_tdb.h"
#include <vector>
#include "conditions.h"
#include <algorithm>
#include <boost/tuple/tuple.hpp>

//      vector(sitefraction[speciesname])
typedef std::vector<std::map<std::string,double>> sublattice_vector;
//      vector(pair<phasename,sublattice(sitefraction[speciesname]))>)
typedef std::vector<std::pair<std::string,sublattice_vector>> sitefracs;
//      tuple<phasename,currentsubl(index 0),species,sitefrac>
typedef boost::tuple<std::string,int,std::string,double> sitefrac_entry;
//      tuple<phasename,currentsubl(index 0),lambda'''>
typedef boost::tuple<std::string,int,double> sitefracbal_entry;
//      tuple<phasename,phasefrac>
typedef boost::tuple<std::string,double> phasefrac_entry;
//      tuple<speciesname,constraint parameter>
typedef boost::tuple<std::string,double> component_entry;

struct vector_map {
	// int,int means begin_index, end_index
	// std::string is name of species
	typedef std::vector<std::pair<int,int>> index_pairs;
	std::vector<boost::tuple<int,int,Phase_Collection::const_iterator>> phasefrac_iters;
	// phase->sublattice->species[name]->pair(index,phase_iter)
	std::vector<std::vector<std::map<std::string,std::pair<int,Phase_Collection::const_iterator>>>> sitefrac_iters;
	index_pairs lambda1p_iters;
	index_pairs lambda2p_iters;
	index_pairs lambda3p_iters;
	index_pairs eta1p_iters;
	index_pairs eta2p_iters;
	index_pairs tau1p_iters;
	index_pairs tau2p_iters;
};

std::vector<double> initialize_variables(
	const Phase_Collection::const_iterator phase_iter,
	const Phase_Collection::const_iterator phase_end,
	const evalconditions &conditions,
	vector_map &retmap
	);

/*void update_gradient(
	const Phase_Collection::const_iterator phase_iter,
	const Phase_Collection::const_iterator phase_end,
	const vector_map &var_map,
	std::vector &variables,
	std::vector &gradient,
	const evalconditions &conditions
	);*/

/*void calculate_hessian(
	const Phase_Collection::const_iterator phase_iter,
	const Phase_Collection::const_iterator phase_end,
	const vector_map &var_map,
	const std::vector &variables,
	arma::matrix &Hessian,
	const evalconditions &conditions
	);*/

double mole_fraction(
	const std::string &spec_name,
	const Sublattice_Collection::const_iterator ref_subl_iter_start,
	const Sublattice_Collection::const_iterator ref_subl_iter_end,
	const sublattice_vector::const_iterator subl_iter_start,
	const sublattice_vector::const_iterator subl_iter_end
	);

double mole_fraction_deriv(
	const std::string &spec_name,
	const std::string &deriv_spec_name,
	const int &deriv_subl_index,
	const Sublattice_Collection::const_iterator ref_subl_iter_start,
	const Sublattice_Collection::const_iterator ref_subl_iter_end,
	const sublattice_vector::const_iterator subl_iter_start,
	const sublattice_vector::const_iterator subl_iter_end
	);

double get_parameter
	(
	const Phase_Collection::const_iterator phase_iter,
	const evalconditions &conditions,
	const std::vector<std::vector<std::string>> &subl_config,
	const std::string type = "G"
	);

double get_Gibbs
	(
	const sublattice_vector::const_iterator subl_start,
	const sublattice_vector::const_iterator subl_end,
	const Phase_Collection::const_iterator phase_iter,
	const evalconditions &conditions
	);

double get_Gibbs_deriv
	(
	const sublattice_vector::const_iterator subl_start,
	const sublattice_vector::const_iterator subl_end,
	const Phase_Collection::const_iterator phase_iter,
	const evalconditions &conditions,
	const int &sublindex,
	const std::string &specname
	);

double multiply_site_fractions
	(
	const sublattice_vector::const_iterator subl_start,
	const sublattice_vector::const_iterator subl_end,
	const Phase_Collection::const_iterator phase_iter,
	const evalconditions &conditions,
	std::vector<std::vector<std::string>> species  = std::vector<std::vector<std::string>>()
	);

double multiply_site_fractions_deriv
	(
	const sublattice_vector::const_iterator subl_start,
	const sublattice_vector::const_iterator subl_end,
	const Phase_Collection::const_iterator phase_iter,
	const evalconditions &conditions,
	const int &sublindex,
	const std::string &specname,
	std::vector<std::vector<std::string>> species = std::vector<std::vector<std::string>>()
	);

#endif