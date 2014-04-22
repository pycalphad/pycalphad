/*=============================================================================
	Copyright (c) 2012-2014 Richard Otis

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
#include "libgibbs/include/optimizer/utils/ndsimplex.hpp"
#include "libgibbs/include/optimizer/utils/convex_hull.hpp"
#include "libgibbs/include/utils/cholesky.hpp"
#include <boost/bimap.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <algorithm>
#include <string>
#include <map>
#include <limits>

std::vector<std::vector<double>> AdaptiveSearchND (
                                  CompositionSet const &phase,
                                  evalconditions const& conditions,
                                  const SimplexCollection &search_region,
                                  const std::size_t depth,
                                  const double old_gradient_mag = 1e12 );

namespace Optimizer
{

// TODO: Should this be a member function of GibbsOpt?
// The function calling LocateMinima definitely should be at least (needs access to all CompositionSets)
// LocateMinima finds all of the minima for a given phase's Gibbs energy
// In addition to allowing us to choose a better starting point, this will allow for automatic miscibility gap detection
std::vector<std::map<std::string,double>>  LocateMinima (
        CompositionSet const &phase,
        sublattice_set const &sublset,
        evalconditions const& conditions,
        const std::size_t depth // depth tracking for recursion
                                       )
{
    // This is the initial amount of subdivision
    constexpr const std::size_t subdivisions_per_axis = 20; // TODO: make this user-configurable
    // minimum edge length of a candidate tie hyperplane
    const double critical_edge_length = 2.0 / (double)subdivisions_per_axis;
    using namespace boost::numeric::ublas;

    // EZD Global Minimization (Emelianenko et al., 2006)
    // First: FIND CONCAVITY REGIONS
    std::vector<std::vector<double>> points;
    std::vector<std::vector<double>> unmapped_minima;
    std::vector<std::map<std::string,double>> minima;
    std::set<std::size_t> dependent_dimensions; // vector of indices to dependent variables
    std::vector<SimplexCollection> start_simplices;
    std::vector<SimplexCollection> positive_definite_regions;
    std::vector<SimplexCollection> components_in_sublattice;
    std::size_t current_dependent_dimension = 0; // index of current dependent dimension

    // Get the first sublattice for this phase
    boost::multi_index::index<sublattice_set,phase_subl>::type::iterator ic0,ic1;
    int sublindex = 0;
    ic0 = boost::multi_index::get<phase_subl> ( sublset ).lower_bound ( boost::make_tuple ( phase.name(), sublindex ) );
    ic1 = boost::multi_index::get<phase_subl> ( sublset ).upper_bound ( boost::make_tuple ( phase.name(), sublindex ) );

    // (1) Sample some points on the domain using NDSimplex
    // Because the grid is uniform, we can assume that each point is the center of an N-simplex
    // Determine number of components in each sublattice
    while ( ic0 != ic1 ) {
        const std::size_t number_of_species = std::distance ( ic0,ic1 );
        if ( number_of_species > 0 ) {
            // Last component is dependent dimension
            current_dependent_dimension += (number_of_species-1);
            dependent_dimensions.insert(current_dependent_dimension);
            NDSimplex base ( number_of_species-1 ); // construct the unit (q-1)-simplex
            components_in_sublattice.emplace_back ( base.simplex_subdivide ( subdivisions_per_axis ) );
        }
        // Next sublattice
        ++sublindex;
        ic0 = boost::multi_index::get<phase_subl> ( sublset ).lower_bound ( boost::make_tuple ( phase.name(), sublindex ) );
        ic1 = boost::multi_index::get<phase_subl> ( sublset ).upper_bound ( boost::make_tuple ( phase.name(), sublindex ) );
    }

    // Take all combinations of generated points in each sublattice
    start_simplices = lattice_complex ( components_in_sublattice );


    for ( SimplexCollection& simpcol : start_simplices ) {
        std::cout << "(";
        std::vector<double> pt;
        for ( NDSimplex& simp : simpcol ) {
            std::vector<double> sub_pt = simp.centroid_with_dependent_component();
            pt.insert ( pt.end(),std::make_move_iterator ( sub_pt.begin() ),std::make_move_iterator ( sub_pt.end() ) );
        }
        for ( auto i = pt.begin(); i != pt.end(); ++i ) {
            std::cout << *i;
            if ( std::distance ( i,pt.end() ) > 1 ) {
                std::cout << ",";
            }
        }
        std::cout << ")" << std::endl;
        points.emplace_back ( std::move ( pt ) );
    }

    // (2) Calculate the Lagrangian Hessian for all sampled points
    for ( SimplexCollection& simpcol : start_simplices ) {
        std::vector<double> pt;
        for ( NDSimplex& simp : simpcol ) {
            // Generate the current point from all the simplices in each sublattice
            std::vector<double> sub_pt = simp.centroid_with_dependent_component();
            pt.insert ( pt.end(),std::make_move_iterator ( sub_pt.begin() ),std::make_move_iterator ( sub_pt.end() ) );
        }
        if ( pt.size() == 0 ) {
            continue;    // skip empty (invalid) points
        }
        symmetric_matrix<double, lower> Hessian ( zero_matrix<double> ( pt.size(),pt.size() ) );
        try {
            Hessian = phase.evaluate_objective_hessian_matrix ( conditions, phase.get_variable_map(), pt );
        } catch ( boost::exception &e ) {
            std::cout << boost::diagnostic_information ( e );
            throw;
        } catch ( std::exception &e ) {
            std::cout << e.what();
            throw;
        }
        //std::cout << "Hessian: " << Hessian << std::endl;
        // NOTE: For this calculation we consider only the linear constraints for an isolated phase (e.g., site fraction balances)
        // (3) Save all points for which the Lagrangian Hessian is positive definite in the null space of the constraint gradient matrix
        //        NOTE: This is the projected Hessian method (Nocedal and Wright, 2006, ch. 12.4, p.349)
        //        But how do I choose the Lagrange multipliers for all the constraints? Can I calculate them?
        //        The answer is that, because the constraints are linear, there is no constraint contribution to the Hessian.
        //        That means that the Hessian of the Lagrangian is just the Hessian of the objective function.
        const std::size_t Zcolumns = pt.size() - phase.get_constraints().size();
        // Z is the constraint null space matrix = phase.get_constraint_null_space_matrix()
        //    (a) Set Hproj = transpose(Z)*(L'')*Z
        matrix<double> Hproj ( pt.size(), Zcolumns );
        Hproj = prod ( trans ( phase.get_constraint_null_space_matrix() ),
                       matrix<double> ( prod ( Hessian,phase.get_constraint_null_space_matrix() ) ) );
        //std::cout << "Hproj: " << Hproj << std::endl;
        //    (b) Verify that all diagonal elements of Hproj are strictly positive; if not, remove this point from consideration
        //        NOTE: This is a necessary but not sufficient condition that a matrix be positive definite, and it's easy to check
        //        Reference: Carlen and Carvalho, 2007, p. 148, Eq. 5.12
        //        TODO: Currently not bothering to do this; should perform a test to figure out if there would be a performance increase
        //    (c) Attempt a Cholesky factorization of Hproj; will only succeed if matrix is positive definite
        const bool is_positive_definite = cholesky_factorize ( Hproj );
        //    (d) If it succeeds, save this point; else, discard it
        if ( is_positive_definite ) {
            positive_definite_regions.push_back ( simpcol );
            for ( double i : pt ) {
                std::cout << i << ",";
            }
            std::cout << ":" <<  std::endl;
        }
    }

    // positive_definite_regions is now filled
    // Perform recursive search for minima on each of the identified regions
    for ( const SimplexCollection &simpcol : positive_definite_regions ) {
        // We start at a recursive depth of 2 because we treat LocateMinima as depth == 1
        // This allows our notation for depth to be consistent with Emelianenko et al.
        std::vector<std::vector<double>> region_minima = AdaptiveSearchND ( phase, conditions, simpcol,  2 );

        // Append this region's minima to the list of minima
        unmapped_minima.reserve ( unmapped_minima.size() +region_minima.size() );
        unmapped_minima.insert ( unmapped_minima.end(), std::make_move_iterator ( region_minima.begin() ),  std::make_move_iterator ( region_minima.end() ) );
    }
    std::cout << "CANDIDATE MINIMA" << std::endl;
    for (auto min : unmapped_minima) {
        for (auto coord : min) {
            std::cout << coord << " ";
        }
        std::cout << std::endl;
    }
    
    // Now the convex hull of the phase needs to be found using the unmapped_minima points
    // I cannot simply lift the sites using the magnitude of the point from the origin
    // due to metastable points
    unmapped_minima = details::lower_convex_hull( 
                         unmapped_minima, dependent_dimensions, critical_edge_length 
                                                );

    // We want to map the indices we used back to variable names for the optimizer
    boost::bimap<std::string,int> indexmap = phase.get_variable_map();
    for ( const std::vector<double> &min : unmapped_minima ) {
        std::map<std::string, double> x_point_map; // variable name -> value
        for ( auto it = min.begin(); it != min.end(); ++it ) {
            const int index = std::distance ( min.begin(),it );
            const std::string varname = indexmap.right.at ( index );
            x_point_map[varname] = *it;
        }
        minima.emplace_back ( std::move ( x_point_map ) );
    }

    return minima;
}

// namespace Optimizer
}

// Recursive function for searching composition space for minima
// Input: Simplex that bounds a positive definite search region (SimplexCollection is used for multiple sublattices)
// Input: Recursion depth
// Output: Vector of minimum points
std::vector<std::vector<double>> AdaptiveSearchND (
    CompositionSet const &phase,
    evalconditions const& conditions,
    const SimplexCollection &search_region,
    const std::size_t depth,
    const double old_gradient_mag )
{
    using namespace boost::numeric::ublas;
    typedef boost::numeric::ublas::vector<double> ublas_vector;
    typedef boost::numeric::ublas::matrix<double> ublas_matrix;
    BOOST_ASSERT ( depth > 0 );
    constexpr const double gradient_magnitude_threshold = 1000;
    constexpr const std::size_t subdivisions_per_axis = 2;
    constexpr const std::size_t max_depth = 10;
    std::vector<std::vector<double>> minima;
    std::vector<double> pt;
    double mag = std::numeric_limits<double>::max();

    std::vector<SimplexCollection> simplex_combinations, new_simplices;
    auto chosen_simplex = new_simplices.cend(); // iterator to the current smallest-gradient simplex

    // simplex_subdivide() the simplices in all the active sublattices
    for ( const NDSimplex &simp : search_region ) {
        simplex_combinations.emplace_back ( simp.simplex_subdivide ( subdivisions_per_axis ) );
    }
    // lattice_complex() the result to generate all the combinations in the sublattices
    new_simplices = lattice_complex ( simplex_combinations );

    // new_simplices now contains a vector of SimplexCollections
    // It's a SimplexCollection instead of an NDSimplex because there is one NDSimplex per sublattice
    // The centroids of each NDSimplex are concatenated (with the dependent component) to get the active point
    // TODO: fix to only send the one with the minimum gradient magnitude to the next depth
    // Calculate the gradient for each newly-created simplex
    for ( auto sc = new_simplices.cbegin(); sc != new_simplices.cend(); ++sc ) {
        std::vector<double> pt, raw_gradient;
        double temp_magnitude = 0;

        // Calculate the objective gradient (L') for the centroid of the active simplex
        for ( const NDSimplex& simp : *sc ) {
            // Generate the current point (pt) from all the simplices in each sublattice
            std::vector<double> sub_pt = simp.centroid_with_dependent_component();
            pt.insert ( pt.end(),std::make_move_iterator ( sub_pt.begin() ),std::make_move_iterator ( sub_pt.end() ) );
        }
        raw_gradient = phase.evaluate_internal_objective_gradient ( conditions, &pt[0] );
        // Project the raw gradient into the null space of constraints
        // This will leave only the gradient in the feasible directions
        ublas_vector projected_gradient ( raw_gradient.size() );
        std::move ( raw_gradient.begin(), raw_gradient.end(), projected_gradient.begin() );
        projected_gradient = prod ( phase.get_gradient_projector(), projected_gradient );

        // Calculate magnitude of projected gradient
        temp_magnitude = norm_2 ( projected_gradient );
        // If this is smaller than the known point, switch to this point
        if ( temp_magnitude < mag ) {
            // We have a new candidate minimum
            chosen_simplex = sc;
            mag = temp_magnitude;
        }
    }

    // Now we must decide whether we have arrived at our terminating condition or to subdivide further
    if ( mag < gradient_magnitude_threshold || depth >= max_depth ) {
        // We've hit our terminating condition
        // It may or may not be a minimum, but it's the best we can find here
        std::vector<double> pt;
        double objective;
        for ( const NDSimplex& simp : *chosen_simplex ) {
            // Generate the current point (pt) from all the simplices in each sublattice
            std::vector<double> sub_pt = simp.centroid_with_dependent_component();
            pt.insert ( pt.end(),std::make_move_iterator ( sub_pt.begin() ),std::make_move_iterator ( sub_pt.end() ) );
        }
        // Add the energy of this configuration as the final coordinate
        objective = phase.evaluate_objective ( conditions, phase.get_variable_map(), &pt[0] );
        pt.emplace_back ( objective );
        minima.emplace_back( std::move( pt ) );
        return minima;
    } else {
        // Keep searching for a minimum by subdividing our chosen_simplex
        // We save a lot of time by only subdividing chosen_simplex!
        std::vector<std::vector<double>> recursive_minima = AdaptiveSearchND ( phase, conditions, *chosen_simplex, depth+1, mag );
        // Add the found minima to the list of known minima
        minima.reserve ( minima.size() +recursive_minima.size() );
        minima.insert ( minima.end(), std::make_move_iterator ( recursive_minima.begin() ),  std::make_move_iterator ( recursive_minima.end() ) );
        return minima;
    }
}
