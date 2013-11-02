/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// opt_Gibbs.h -- declaration for Gibbs optimizer class

#ifndef INCLUDED_OPT_GIBBS
#define INCLUDED_OPT_GIBBS

#include "external/coin/IpTNLP.hpp"
#include "libtdb/include/structure.hpp"
#include "libtdb/include/utils/math_expr.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/optimizer.hpp"
#include "libgibbs/include/constraint.hpp"
#include <boost/spirit/include/support_utree.hpp>
#include <string>
#include <unordered_map>
#include <utility>

using namespace Ipopt;
typedef std::map<std::string, int> index_table; // matches variable names to Ipopt indices

class GibbsOpt : public TNLP {
public:
	GibbsOpt(
		const Database &DB,
		const evalconditions &sysstate);
	virtual ~GibbsOpt();
	/**@name Overloaded from TNLP */
	//@{
	/** Method to return some info about the nlp */
	virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
		Index& nnz_h_lag, IndexStyleEnum& index_style);

	/** Method to return the bounds for my problem */
	virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
		Index m, Number* g_l, Number* g_u);

	/** Method to return the starting point for the algorithm */
	virtual bool get_starting_point(Index n, bool init_x, Number* x,
		bool init_z, Number* z_L, Number* z_U,
		Index m, bool init_lambda,
		Number* lambda);

	/** Method to return the objective value */
	virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value);

	/** Method to return the gradient of the objective */
	virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f);

	/** Method to return the constraint residuals */
	virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g);

	/** Method to return:
	*   1) The structure of the jacobian (if "values" is NULL)
	*   2) The values of the jacobian (if "values" is not NULL)
	*/
	virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
		Index m, Index nele_jac, Index* iRow, Index *jCol,
		Number* values);

	/** Method to return:
	*   1) The structure of the hessian of the lagrangian (if "values" is NULL)
	*   2) The values of the hessian of the lagrangian (if "values" is not NULL)
	*/
	virtual bool eval_h(Index n, const Number* x, bool new_x,
		Number obj_factor, Index m, const Number* lambda,
		bool new_lambda, Index nele_hess, Index* iRow,
		Index* jCol, Number* values);

	//@}

	/** @name Solution Methods */
	//@{
	/** This method is called when the algorithm is complete so the TNLP can store/write the solution */
	virtual void finalize_solution(SolverReturn status,
		Index n, const Number* x, const Number* z_L, const Number* z_U,
		Index m, const Number* g, const Number* lambda,
		Number obj_value,
		const IpoptData* ip_data,
		IpoptCalculatedQuantities* ip_cq);
	//@}

	typedef std::unordered_map<std::string, double> speclist; // species name + site fraction
	typedef std::pair<double, speclist> sublattice; // stoichiometric coefficient + species list
	typedef std::vector<sublattice> constitution; // collection of sublattices
	typedef std::pair<double, constitution> phase; // phase fraction + constitution
	typedef std::unordered_map<std::string,phase> phasemap;

	phasemap get_phase_map() { return ph_map; };

private:
	/**@name Methods to block default compiler methods.
	* The compiler automatically generates the following three methods.
	*  Since the default compiler implementation is generally not what
	*  you want (for all but the most simple classes), we usually 
	*  put the declarations of these methods in the private section
	*  and never implement them. This prevents the compiler from
	*  implementing an incorrect "default" behavior without us
	*  knowing. (See Scott Meyers book, "Effective C++")
	*  
	*/
	//@{
	//  GibbsOpt();
	GibbsOpt(const GibbsOpt&);
	GibbsOpt& operator=(const GibbsOpt&);
	//@}
	sublattice_set main_ss;
	vector_map var_map;
	index_table main_indices;
	sitefracs mysitefracs;
	evalconditions conditions;
	boost::spirit::utree master_tree; // abstract syntax tree (AST) for the objective function
	Phase_Collection::const_iterator phase_iter;
	Phase_Collection::const_iterator phase_end;
	Phase_Collection phase_col;
	ConstraintManager cm;
	std::map<int,boost::spirit::utree> first_derivatives;
	std::vector<jacobian_entry> jac_g_trees;
	hessian_set hessian_data;
	std::vector<int> fixed_indices;

	// data structure for final result
	phasemap ph_map; // maps phase name to its object (final result)
};

#endif
