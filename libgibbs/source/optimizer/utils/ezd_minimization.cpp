/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// subroutines for EZD global minimization
// Reference: Maria Emelianenko, Zi-Kui Liu, and Qiang Du.
// "A new algorithm for the automation of phase diagram calculation." Computational Materials Science 35.1 (2006): 61-74.

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/utils/ezd_minimization.hpp"
#include "libgibbs/include/compositionset.hpp"
#include "libgibbs/include/constraint.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/halton.hpp"
#include "libgibbs/include/optimizer/utils/ndgrid.hpp"
#include "libgibbs/include/utils/cholesky.hpp"
#include "libgibbs/include/utils/qr.hpp"
#include <boost/bimap.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <algorithm>
#include <string>
#include <map>

namespace Optimizer {

// TODO: Should this be a member function of GibbsOpt?
// The function calling LocateMinima definitely should be at least (needs access to all CompositionSets)
// LocateMinima finds all of the minima for a given phase's Gibbs energy
// In addition to allowing us to choose a better starting point, this will allow for automatic miscibility gap detection
void LocateMinima(
		CompositionSet const &phase,
		sublattice_set const &sublset,
		evalconditions const& conditions,
		const std::size_t depth // depth tracking for recursion
		) {
	constexpr const std::size_t grid_points_per_axis = 10; // TODO: make this user-configurable
	using namespace boost::numeric::ublas;
	// Because the grid is uniform, we can assume that each point is the center of an N-cube
	// of width max_extent-min_extent. Boundary points are a special case.
	// Drop points outside the feasible region.
	//
	// EZD Global Minimization (Emelianenko et al., 2006)
	// For depth = 1: FIND CONCAVITY REGIONS
	if (depth == 1) {
		std::vector<std::vector<double> > points;

		// Get all the sublattices for this phase
		boost::multi_index::index<sublattice_set,phase_subl>::type::iterator ic0,ic1;
		ic0 = boost::multi_index::get<phase_subl>(sublset).lower_bound(boost::make_tuple(phase.name(), 0));
		ic1 = boost::multi_index::get<phase_subl>(sublset).end();

		// (1) Sample some points on the domain using NDGrid
		// TODO: This is going to generate a LOT of infeasible points. Is rejection sampling sufficient?
		// A better(?) implementation would handle the sublattice internal degrees of freedom separately
		auto point_add = [&points,&phase,&conditions](std::vector<double> &address) {
			// Determine if the point is feasible
			double constraint_violation = 0;
			constexpr const double max_constraint_violation = 1e-4; // TODO: make user-configurable
			for (auto i = phase.get_constraints().cbegin(); i != phase.get_constraints().cend(); ++i) {
				double lhs =
						process_utree(i->lhs, conditions, phase.get_variable_map(), const_cast<double*>(&address[0])).get<double>();
				double rhs =
						process_utree(i->rhs, conditions, phase.get_variable_map(), const_cast<double*>(&address[0])).get<double>();
				constraint_violation += fabs(rhs-lhs);
				if (constraint_violation > max_constraint_violation) return; // point is infeasible; skip
			}
			std::cout << "adding point [";
			for (auto i = address.begin(); i != address.end(); ++i) {
				std::cout << *i;
				if (std::distance(i,address.end()) > 1) std::cout << ",";
			}
			std::cout << "]" << std::endl;
			points.push_back(address);
		};
		NDGrid::sample(0, 1, std::distance(ic0,ic1), grid_points_per_axis, point_add);

		// (2) Calculate the Lagrangian Hessian for all sampled points
		for (auto pt : points) {
			symmetric_matrix<double, lower> Hessian(zero_matrix<double>(pt.size(),pt.size()));
			try {
				Hessian = phase.evaluate_objective_hessian_matrix(conditions, phase.get_variable_map(), pt);
			}
			catch (boost::exception &e) {
				std::cout << boost::diagnostic_information(e);
				throw;
			}
			catch (std::exception &e) {
				std::cout << e.what();
				throw;
			}
			std::cout << "Hessian: " << Hessian << std::endl;
			// NOTE: For this calculation we consider only the linear constraints for an isolated phase (e.g., site fraction balances)
			// (3) Save all points for which the Lagrangian Hessian is positive definite in the null space of the constraint gradient matrix
			//        NOTE: This is the projected Hessian method (Nocedal and Wright, 2006, ch. 12.4, p.349)
			//        But how do I choose the Lagrange multipliers for all the constraints? Can I calculate them?
			//        The answer is that, because the constraints are linear, there is no constraint contribution to the Hessian.
			//        That means that the Hessian of the Lagrangian is just the Hessian of the objective function.
			//        n = number of variables, m = number of active constraints
			//    (a) Form matrix trans(A) (n x m), the transpose of the Jacobian of active constraints (constraint gradient matrix)
			matrix<double> transA(zero_matrix<double>(pt.size(),phase.get_constraints().size()));
			for (auto j : phase.get_jacobian()) // Calculate transpose of Jacobian
				transA(j.var_index,j.cons_index) =
						process_utree(j.ast, conditions, phase.get_variable_map(), const_cast<double*>(&pt[0])).get<double>();
			std::cout << "transA: " << transA << std::endl;
			//    (b) Compute the full QR decomposition of transpose(A)
			std::vector<double> betas = inplace_qr(transA);
			matrix<double> Q(zero_matrix<double>(pt.size(),pt.size()));
			matrix<double> R(zero_matrix<double>(pt.size(),phase.get_constraints().size()));
			recoverQ(transA, betas, Q, R);
			std::cout << "Q: " << Q << std::endl;
			std::cout << "R: " << R << std::endl;
			//    (c) Copy the last n-m columns of Q into Z (related to the bottom n-m rows of R which should all be zero)
			//        Reference: Eq. 12.71, p. 349 of Nocedal and Wright, 2006
			const std::size_t Zcolumns = pt.size() - phase.get_constraints().size();
			matrix<double> Z(pt.size(), Zcolumns);
			// Z is the submatrix of Q that includes all of Q's rows and its rightmost n-m columns
			Z = subrange(Q, 0,pt.size(), phase.get_constraints().size(),pt.size());
			std::cout << "Z: " << Z << std::endl;
			//    (d) Set Hproj = transpose(Z)*(L'')*Z
			matrix<double> Hproj(pt.size(), Zcolumns);
			Hproj = prod(trans(Z), matrix<double>(prod(Hessian,Z)));
			std::cout << "Hproj: " << Hproj << std::endl;
			//    (e) Verify that all diagonal elements of Hproj are strictly positive; if not, remove this point from consideration
			//        NOTE: This is a necessary but not sufficient condition that a matrix be positive definite, and it's easy to check
			//        Reference: Carlen and Carvalho, 2007, p. 148, Eq. 5.12
			//    (f) Attempt a Cholesky factorization of Hproj; will only succeed if matrix is positive definite
			const bool is_positive_definite = cholesky_factorize(Hessian);
			//    (g) If it succeeds, save this point; else, remove it
			if (is_positive_definite) {
				std::cout << "CONVEX FEASIBLE: ";
				std::cout << "adding point [";
				for (auto i = pt.begin(); i != pt.end(); ++i) {
					std::cout << *i;
					if (std::distance(i,pt.end()) > 1) std::cout << ",";
				}
				std::cout << "]" << std::endl;
			}
			else {

			}
			// (4) For each saved point, send to next depth
		}
	}
	// For depth > 1: FIND MINIMA
	else if (depth > 1) {
		// (1) Sample some points on the domain using NDGrid (probably separate function)
		// (2) Calculate the Lagrangian gradient (L') for all sampled points
		// (3) Find the point (z) with the minimum magnitude of L'
		// (4) If that magnitude is less than some defined epsilon, or we've hit max_depth, return z as a minimum
		//     Else use N-cube assumption to define new domain around z and send to next depth (return minima from that)
	}
	else {
		// invalid depth; throw exception
	}
}

}
