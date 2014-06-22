/*=============================================================================
	Copyright (c) 2012-2014 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// subroutines for modified EZD global minimization
// Reference for original EZD: Maria Emelianenko, Zi-Kui Liu, and Qiang Du.
// "A new algorithm for the automation of phase diagram calculation." Computational Materials Science 35.1 (2006): 61-74.

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/utils/ezd_minimization.hpp"
#include "libgibbs/include/compositionset.hpp"
#include "libgibbs/include/constraint.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/halton.hpp"
#include "libgibbs/include/optimizer/utils/hull_mapping.hpp"
#include "libgibbs/include/optimizer/utils/ndsimplex.hpp"
#include "libgibbs/include/optimizer/utils/convex_hull.hpp"
#include "libgibbs/include/utils/cholesky.hpp"
#include "libgibbs/include/utils/site_fraction_convert.hpp"
#include "libtdb/include/exceptions.hpp"
#include <libqhullcpp/QhullFacet.h>
#include <libqhullcpp/QhullFacetList.h>
#include <boost/bimap.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <algorithm>
#include <string>
#include <map>
#include <limits>

namespace Optimizer { namespace details {

std::vector<std::vector<double>> AdaptiveSearchND (
                                  CompositionSet const &phase,
                                  evalconditions const& conditions,
                                  const SimplexCollection &search_region,
                                  const std::size_t refinement_subdivisions_per_axis,
                                  const std::size_t depth,
                                  const double old_gradient_mag = 1e12 );

std::vector<double> generate_point ( const SimplexCollection &simpcol );


// TODO: Should this be a member function of GibbsOpt?
// The function calling LocateMinima definitely should be at least (needs access to all CompositionSets)
// LocateMinima finds all of the minima for a given phase's Gibbs energy
// In addition to allowing us to choose a better starting point, this will allow for automatic miscibility gap detection
std::vector<std::vector<double>>  AdaptiveSimplexSample (
        CompositionSet const &phase,
        sublattice_set const &sublset,
        evalconditions const& conditions,
        const std::size_t initial_subdivisions_per_axis,
        const std::size_t refinement_subdivisions_per_axis,
        const bool discard_unstable
        )
{
    using namespace boost::numeric::ublas;

    // EZD Global Minimization (Emelianenko et al., 2006)
    // First: FIND CONCAVITY REGIONS
    std::vector<std::vector<double>> points;
    std::vector<std::vector<double>> unmapped_minima;
    std::vector<SimplexCollection> start_simplices;
    std::vector<SimplexCollection> positive_definite_regions;
    std::vector<SimplexCollection> components_in_sublattice;
    std::vector<std::vector<std::vector<double>>> pure_end_members, all_permutations;
    
    // Simplified lambda for energy calculation
    auto calculate_energy = [&phase,&conditions] (const std::vector<double>& point) {
        return phase.evaluate_objective(conditions,phase.get_variable_map(),const_cast<double*>(&point[0]));
    };

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
        BOOST_ASSERT ( number_of_species > 0 );
        NDSimplex base ( number_of_species-1 ); // construct the unit (q-1)-simplex
        components_in_sublattice.emplace_back ( base.simplex_subdivide ( initial_subdivisions_per_axis ) );
        // Next sublattice
        ++sublindex;
        ic0 = boost::multi_index::get<phase_subl> ( sublset ).lower_bound ( boost::make_tuple ( phase.name(), sublindex ) );
        ic1 = boost::multi_index::get<phase_subl> ( sublset ).upper_bound ( boost::make_tuple ( phase.name(), sublindex ) );
    }

    // Take all combinations of generated points in each sublattice
    start_simplices = lattice_complex ( components_in_sublattice );


    for ( SimplexCollection& simpcol : start_simplices ) {
        std::cout << "(";
        std::vector<double> pt = generate_point ( simpcol );
        for ( auto i = pt.begin(); i != pt.end(); ++i ) {
            std::cout << *i;
            if ( std::distance ( i,pt.end() ) > 1 ) {
                std::cout << ",";
            }
        }
        std::cout << ")" << std::endl;
        points.emplace_back ( std::move ( pt ) );
    }

    if (discard_unstable) {
        // (2) Calculate the Lagrangian Hessian for all sampled points
        for ( SimplexCollection& simpcol : start_simplices ) {
            std::vector<double> pt = generate_point ( simpcol );
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
                positive_definite_regions.emplace_back ( simpcol );
                /*DEBUG for ( double i : pt ) {
                    std::cout << i << ",";
                }
                std::cout << ":" <<  std::endl;*/
            }
        }
    }
    else {
        // Save all points (do not discard unstable regions)
        positive_definite_regions = start_simplices;
    }
    
    // The pure end-members are always considered in the calculation, so add them
    // This will handle the case of complete immiscibility: energy function is nonconvex
    sublindex = 0;
    ic0 = boost::multi_index::get<phase_subl> ( sublset ).lower_bound ( boost::make_tuple ( phase.name(), sublindex ) );
    ic1 = boost::multi_index::get<phase_subl> ( sublset ).upper_bound ( boost::make_tuple ( phase.name(), sublindex ) );
    while ( ic0 != ic1 ) {
        const std::size_t number_of_species = std::distance ( ic0,ic1 );
        BOOST_ASSERT ( number_of_species > 0 );
        std::vector<std::vector<double>> sublattice_permutations;
        const double epsilon_composition = 1e-12;
        std::vector<double> sub_pt ( number_of_species, epsilon_composition );
        sub_pt[number_of_species-1] = 1-(number_of_species-1)*epsilon_composition;
        sublattice_permutations.push_back ( sub_pt );
        // Here we take advantage of the fact that sub_pt is sorted by construction
        // We will iterate from (0,0,...,1) to (1,0,...,0)
        while ( std::next_permutation ( sub_pt.begin(), sub_pt.end() ) ) {
            sublattice_permutations.push_back ( sub_pt );
        }
        all_permutations.emplace_back ( sublattice_permutations );
        // Next sublattice
        ++sublindex;
        ic0 = boost::multi_index::get<phase_subl> ( sublset ).lower_bound ( boost::make_tuple ( phase.name(), sublindex ) );
        ic1 = boost::multi_index::get<phase_subl> ( sublset ).upper_bound ( boost::make_tuple ( phase.name(), sublindex ) );
    }
    // Take all combinations of generated points in each sublattice
    pure_end_members = lattice_complex ( all_permutations );
    for ( auto &pure_points : pure_end_members ) {
        // We need to concatenate all the sublattice coordinates in pure_points
        std::vector<double> pt;
        for ( auto &coords : pure_points ) {
            if (coords.size() == 0) continue;
            pt.reserve ( pt.size() + coords.size() );
            pt.insert ( pt.end(), std::make_move_iterator ( coords.begin() ),  std::make_move_iterator ( coords.end() ) );
        }
        std::cout << "checking ";
        for ( auto i = pt.begin(); i != pt.end(); ++i ) {
            std::cout << *i;
            if ( std::distance ( i,pt.end() ) > 1 ) {
                std::cout << ",";
            }
        }
        std::cout << std::endl;
        // Before convex_hull, unmapped_minima has an energy coordinate
        pt.push_back ( calculate_energy ( pt ) );
        std::cout << "ENDMEMBER ";
        for ( auto & coord : pt ) {
            std::cout << coord << ",";
        }
        std::cout << std::endl;
        unmapped_minima.emplace_back ( std::move ( pt ) );
    }
    // If no unstable regions were found, there's no point in continuing the search
    if ( start_simplices.size() == positive_definite_regions.size() ) {
        // copy the unrefined grid into the return value
        for ( auto simp_iter = start_simplices.begin(); simp_iter != start_simplices.end(); ++simp_iter ) {
            auto gridpoint = generate_point ( *simp_iter );
            gridpoint.push_back ( calculate_energy ( gridpoint ) );
            unmapped_minima.emplace_back ( gridpoint );
        }
    }
    else if ( positive_definite_regions.size() > 0 ) {
        // positive_definite_regions is now filled
        // At least one unstable region was found
        // Perform recursive search for minima on each of the identified regions
        for ( const SimplexCollection &simpcol : positive_definite_regions ) {
            //std::cout << "checking simplexcollection of size " << simpcol.size() << std::endl;
            std::vector<std::vector<double>> region_minima = AdaptiveSearchND ( phase, conditions, simpcol, refinement_subdivisions_per_axis, 1 );

            // Append this region's minima to the list of minima
            unmapped_minima.reserve ( unmapped_minima.size() +region_minima.size() );
            unmapped_minima.insert ( unmapped_minima.end(), std::make_move_iterator ( region_minima.begin() ),  std::make_move_iterator ( region_minima.end() ) );
        }
        /*DEBUG std::cout << "CANDIDATE MINIMA" << std::endl;
        for (auto min : unmapped_minima) {
            for (auto coord : min) {
                std::cout << coord << " ";
            }
            std::cout << std::endl;
        }*/
    }
    return unmapped_minima;

    // TODO: Apply phase-specific user-supplied constraints to the system
    // TODO: Map to mole fraction space
    /*
    std::cout << "MAPPING MINIMA" << std::endl;
    for ( auto minima = unmapped_minima.begin(); minima != unmapped_minima.end(); ++minima ) {
        const std::size_t minima_index = std::distance ( unmapped_minima.begin(), minima );
        const double point_energy = minima->back(); // last coordinate is energy
        minima->pop_back();
        for ( auto &coord : *minima ) {
            std::cout << coord << ",";
        }
        std::cout << ":";
        std::map<std::string,double> mapped_minima = convert_site_fractions_to_mole_fractions ( phase.name(), sublset, *minima );
        for ( auto &coord : mapped_minima ) {
            std::cout << coord.first << "->" << coord.second << ",";
        }
        std::cout << std::endl;
    }*/
    
    // TODO: Apply known conditions
    
    /* Design notes for constrained global minimization
     * (1)  Get the points from the lower_convex_hull() of this phase.
     * (2)  Apply any user-supplied conditions related to the internal degrees of freedom.
     * (3)  Map the remaining facet vertices to the global mole fraction space.
     *      (a) Each point must somehow be associated with its original internal degrees of freedom.
     * (4)  Re-run Qhull to calculate the lower convex hull in this space.
     * (5)  Apply any user-supplied constraints for this phase's state space (activity,composition,etc.)
     * (6)  For each vertex of the candidate tie hyperplane, recall its internal degrees of freedom.
     * (7)  Return a vector of these points to the calling function. Each point represents a composition set.
     *      (a) Instead of returning just a point vector, return a QhullPointSet with the
     *      mole fraction points still associated with their internal degrees of freedom.
     *      (b) Combine all candidate points from all phases and calculate the lower convex hull.
     *      (c) Apply user-supplied conditions related to global state space (activity,composition,etc.)
     * (8)  If everything was done right, exactly one facet, the candidate hyperplane, will remain.
     *      (a) Throw if this is not true. If zero, overconstrained. If greater than one, underconstrained.
     *      (b) It would be nice to report or track the number of facets remaining after applying each
     *            constraint.
     * (9)  Use the internal degrees of freedom associated with each vertex to set the phase composition.
     * (10) Find the equilibrium point on the tie plane by applying the user-supplied conditions 
     *      for the global state space (activity, composition, etc).
     *      (a) Fix composition: Set that coordinate to that value.
     *      (b) Fix activity
     *          (i)  Use Qhull option QGn to require visible from point n on pure component axis.
     *          (ii) Use Qhull QVn to add point n to convex hull.
     *          (iii) Keep facets with negative energy orientation (normal points down). (Pdk:n)
     *      (c) Dependent composition is subtract the sum of the rest from 1
     * (11) Return the overall starting point. Metastable phases will be discarded for now.
     */

    // We want to map the indices we used back to variable names for the optimizer
    /*
    boost::bimap<std::string,int> indexmap = phase.get_variable_map();
    for ( const std::vector<double> &min : unmapped_minima ) {
        std::map<std::string, double> x_point_map; // variable name -> value
        for ( auto it = min.begin(); it != min.end(); ++it ) {
            const int index = std::distance ( min.begin(),it );
            const std::string varname = indexmap.right.at ( index );
            x_point_map[varname] = *it;
        }
        minima.emplace_back ( std::move ( x_point_map ) );
    }*/
}

// Recursive function for searching composition space for minima
// Input: Simplex that bounds a positive definite search region (SimplexCollection is used for multiple sublattices)
// Input: Recursion depth
// Output: Vector of minimum points
std::vector<std::vector<double>> AdaptiveSearchND (
    CompositionSet const &phase,
    evalconditions const& conditions,
    const SimplexCollection &search_region,
    const std::size_t refinement_subdivisions_per_axis,
    const std::size_t depth,
    const double old_gradient_mag )
{
    using namespace boost::numeric::ublas;
    typedef boost::numeric::ublas::vector<double> ublas_vector;
    typedef boost::numeric::ublas::matrix<double> ublas_matrix;
    BOOST_ASSERT ( depth > 0 );
    constexpr const double gradient_magnitude_threshold = 1000;
    constexpr const std::size_t max_depth = 5;
    std::vector<std::vector<double>> minima;
    double mag = std::numeric_limits<double>::max();

    std::vector<SimplexCollection> simplex_combinations, new_simplices;
    auto chosen_simplex = new_simplices.cend(); // iterator to the current smallest-gradient simplex

    // simplex_subdivide() the simplices in all the active sublattices
    for ( const NDSimplex &simp : search_region ) {
        simplex_combinations.emplace_back ( simp.simplex_subdivide ( refinement_subdivisions_per_axis ) );
    }
    // lattice_complex() the result to generate all the combinations in the sublattices
    new_simplices = lattice_complex ( simplex_combinations );

    // new_simplices now contains a vector of SimplexCollections
    // It's a SimplexCollection instead of an NDSimplex because there is one NDSimplex per sublattice
    // The centroids of each NDSimplex are concatenated (with the dependent component) to get the active point
    // Calculate the gradient for each newly-created simplex
    for ( auto sc = new_simplices.cbegin(); sc != new_simplices.cend(); ++sc ) {
        std::vector<double> pt = generate_point ( *sc );
        std::vector<double> raw_gradient;
        double temp_magnitude = 0;
        double objective = phase.evaluate_objective ( conditions, phase.get_variable_map(), &pt[0] );
        // Calculate the objective gradient (L') for the centroid of the active simplex
        raw_gradient = phase.evaluate_internal_objective_gradient ( conditions, &pt[0] );
        // Project the raw gradient into the null space of constraints
        // This will leave only the gradient in the feasible directions
        ublas_vector projected_gradient ( raw_gradient.size() );
        std::move ( raw_gradient.begin(), raw_gradient.end(), projected_gradient.begin() );
        //std::cout << "phase.get_gradient_projector.size1() = " << phase.get_gradient_projector().size1() << std::endl;
        projected_gradient = prod ( phase.get_gradient_projector(), projected_gradient );
        //std::cout << "projected_gradient.size() = " << projected_gradient.size() << std::endl;

        // Calculate magnitude of projected gradient
        temp_magnitude = norm_2 ( projected_gradient );
        //std::cout << temp_magnitude << std::endl;
        // If this is smaller than the known point, switch to this point
        if ( temp_magnitude < mag ) {
            // We have a new candidate minimum
            chosen_simplex = sc;
            mag = temp_magnitude;
        }
        // Add every point to mesh, with energy coordinate
        pt.emplace_back ( objective );
        minima.emplace_back( std::move( pt ) );
    }
    
    const bool poor_progress = ( mag > 5000 ) && ( old_gradient_mag > 5000 ) && ( depth > 4 );
    // Now we must decide whether we have arrived at our terminating condition or to subdivide further
    if ( mag < gradient_magnitude_threshold || depth >= max_depth || poor_progress ) {
        // We've hit our terminating condition
        // It may or may not be a minimum, but it's the best we can find here
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

// Generate the full coordinates (with dependent dimensions) from the SimplexCollection of a region
std::vector<double> generate_point ( const SimplexCollection &simpcol ) {
    std::vector<double> pt;
    for ( const NDSimplex& simp : simpcol ) {
        // Generate the current point (pt) from all the simplices in each sublattice
        std::vector<double> sub_pt = simp.centroid_with_dependent_component();
        pt.insert ( pt.end(),std::make_move_iterator ( sub_pt.begin() ),std::make_move_iterator ( sub_pt.end() ) );
    }
    return pt;
}

} // namespace details
} // namespace Optimizer

