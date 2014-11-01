/*=============================================================================
  Copyright (c) 2012-2014 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// N-Dimensional Simplex Point Generation

#ifndef INCLUDED_NDSIMPLEX
#define INCLUDED_NDSIMPLEX

#include "libgibbs/include/optimizer/halton.hpp"
#include "libgibbs/include/optimizer/utils/ndsimplex_colorscheme.hpp"
#include "libgibbs/include/utils/primes.hpp"
#include <boost/assert.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>
#include <cmath>


class NDSimplex
{
public:
  typedef boost::numeric::ublas::matrix<std::size_t> ColorMatrixType;
  typedef boost::numeric::ublas::matrix<double> SimplexMatrixType;
  NDSimplex ( ) { } // default ctor required for Boost Graph Library
  // If we only know the dimension, we'll initialize the unit simplex
  NDSimplex ( const std::size_t dim )
  {
    using namespace boost::numeric::ublas;
    if ( dim > 0 ) {
        // First column is zero; subsequent columns are from the dim x dim identity matrix
        this->vertices = zero_matrix<double> ( dim, dim+1 );
        matrix_range<boost::numeric::ublas::matrix<double>> vr ( this->vertices, range ( 0, this->vertices.size1() ),  range ( 1, this->vertices.size2() ) );
        vr = identity_matrix<double> ( dim );
    }
    else {
        // a null simplex
        this->vertices = identity_matrix<double> ( 1,1 );
    }
  }
  // Construct a simplex from its vertices
  NDSimplex ( SimplexMatrixType &simplexmat )
  {
    BOOST_ASSERT ( simplexmat.size2() - simplexmat.size1() == 1 ); // dimensionality check
    this->vertices = std::move ( simplexmat );
  }
  // Perform the k^dim subdivision of the current simplex; return all of the subsimplices
  std::vector<NDSimplex> simplex_subdivide ( const std::size_t k ) const
  {
    using namespace Optimizer;
    using namespace boost::numeric::ublas;
    const std::size_t d = vertices.size1();
    std::vector<NDSimplex> retvec;
    retvec.reserve ( std::pow ( k, d ) );
    // can't subdivide null simplex: just return itself
    if ( vertices.size1() == vertices.size2() == 1 ) {
        retvec.push_back ( *this );
        return retvec;
    }
    std::vector<ColorMatrixType> colorschemes = generate_color_schemes ( k, d );
    for ( auto i = colorschemes.begin(); i != colorschemes.end(); ++i )
      {
        matrix<double> simplex_coords = zero_matrix<double> ( d,d+1 );
        for ( auto j = 0; j < d+1; ++j )
          {
            matrix_column<matrix<double>> p ( simplex_coords,j );
            for ( auto m = 0; m < k; ++m )
              {
                const matrix_column<const matrix<double>> s ( this->vertices, ( *i ) ( m,j ) );
                p += ( s / ( double ) k );
              }
          }
        retvec.emplace_back ( simplex_coords );
      }
    return retvec;
  }
  // Return the centroid of the simplex
  boost::numeric::ublas::vector<double> centroid() const {
   using namespace boost::numeric::ublas;
   boost::numeric::ublas::vector<double> cent = boost::numeric::ublas::zero_vector<double>(vertices.size1());
   for ( auto j = 0; j < vertices.size2(); ++j )
   {
    const matrix_column<const matrix<double>> p ( vertices,j );
    cent += p;
   }
   cent /= (double)vertices.size2();
   return cent;
  }
  // Return the vector of size d+1 where the first d components are the centroid, and the last is 1 - sum of coordinates
  std::vector<double> centroid_with_dependent_component() const {
	  boost::numeric::ublas::vector<double> cent = this->centroid();
	  std::vector<double> mypoint;
	  mypoint.reserve(cent.size()+1);
	  mypoint.assign(cent.begin(),cent.end()); // Copy in the coordinates
          if (vertices.size1() == vertices.size2() == 1) {
              // null simplex: don't add the dependent coordinate because we're storing it
          }
          else {
              // Calculate 1 - sum(cent), which is the value of the dependent component
              double pointsum = std::accumulate (cent.begin(), 
                                                 cent.end(), 1.0, std::minus<double>());
              mypoint.push_back(pointsum); // Add the dependent component
          }
	  return mypoint;
  }
  const std::size_t dimension() const {
      if (vertices.size1() == vertices.size2() == 1) return 0;
      else return vertices.size1(); 
  }
  const SimplexMatrixType get_vertices() const { return vertices; }
private:
	SimplexMatrixType vertices;                         // vertices of the simplex; each column is a coordinate
};

typedef std::vector<NDSimplex> SimplexCollection;
  // Reference: Chasalow and Brand, 1995, "Algorithm AS 299: Generation of Simplex Lattice Points"
  /*template <typename Func> static inline void lattice (
    const std::size_t point_dimension,
    const std::size_t grid_points_per_major_axis,
    const Func &func
  )
  {
    BOOST_ASSERT ( grid_points_per_major_axis >= 2 );
    BOOST_ASSERT ( point_dimension >= 1 );
    typedef std::vector<double> PointType;
    const double lattice_spacing = 1.0 / ( double ) grid_points_per_major_axis;
    const PointType::value_type lower_limit = 0;
    const PointType::value_type upper_limit = 1;
    PointType point; // Contains current point
    PointType::iterator coord_find; // corresponds to 'j' in Chasalow and Brand

    // Special case: 1 component; only valid point is {1}
    if ( point_dimension == 1 )
      {
        point.push_back ( 1 );
        func ( point );
        return;
      }

    // Initialize algorithm
    point.resize ( point_dimension, lower_limit ); // Fill with smallest value (0)
    const PointType::iterator last_coord = --point.end();
    coord_find = point.begin();
    *coord_find = upper_limit;
    // point should now be {1,0,0,0....}

    do
      {
        func ( point );
        *coord_find -= lattice_spacing;
        if ( *coord_find < lattice_spacing/2 ) *coord_find = lower_limit; // workaround for floating point issues
        if ( std::distance ( coord_find,point.end() ) > 2 )
          {
            ++coord_find;
            *coord_find = lattice_spacing + *last_coord;
            *last_coord = lower_limit;
          }
        else
          {
            *last_coord += lattice_spacing;
            while ( *coord_find == lower_limit ) --coord_find;
          }
      }
    while ( *last_coord < upper_limit );

    func ( point ); // should be {0,0,...1}
  }*/

  // Input: Vector of (vector of objects in each sublattice)
  // Output: All combinations of those vectors
  template <typename T>
  std::vector<std::vector<T>> lattice_complex (const std::vector<std::vector<T>> &components_in_sublattices)
  {
    std::vector<std::vector<T>> point_lattices; // Objects from each sublattice
    std::vector<std::vector<T>> points; // The final return points (all combinations from each vector)
    std::size_t expected_points = 1;
    std::size_t point_dimension = components_in_sublattices.size();

    for ( auto i = components_in_sublattices.cbegin(); i != components_in_sublattices.cend(); ++i )
      {
        expected_points *= i->size();
        point_lattices.push_back ( *i ); // push points for each simplex
      }

    points.reserve ( expected_points );
    
    for ( auto p = 0; p < expected_points; ++p )
      {
        std::vector<T> point;
        std::size_t dividend = p;
        point.reserve ( point_dimension );
        //std::cout << "p : " << p << " indices: [";
        for ( auto r = point_lattices.rbegin(); r != point_lattices.rend(); ++r )
          {
            //std::cout << dividend % r->size() << ",";
            point.insert ( point.end(), ( *r ) [dividend % r->size()] );
            dividend = dividend / r->size();
          }
        //std::cout << "]" << std::endl;
        std::reverse ( point.begin(),point.end() );
        points.push_back ( point );
      }

    return points;
  }

  // Reference for Halton sequence: Hess and Polak, 2003.
  // Reference for uniformly sampling the simplex: Any text on the Dirichlet distribution
  template <typename Func> void quasirandom_sample (
    const std::size_t point_dimension,
    const std::size_t number_of_points,
    const Func &func
  )
  {
    BOOST_ASSERT ( point_dimension < primes_size() ); // No realistic problem should ever violate this
    // TODO: Add the shuffling part to the Halton sequence. This will help with correlation problems for large N
    // TODO: Default-add the end-members (vertices) of the N-simplex
    for ( auto sequence_pos = 1; sequence_pos <= number_of_points; ++sequence_pos )
      {
        std::vector<double> point;
        double point_sum = 0;
        for ( auto i = 0; i < point_dimension; ++i )
          {
            // Draw the coordinate from an exponential distribution
            // N samples from the exponential distribution, when normalized to 1, will be distributed uniformly
            // on a facet of the N-simplex.
            // If X is uniformly distributed, then -LN(X) is exponentially distributed.
            // Since the Halton sequence is a low-discrepancy sequence over [0,1], we substitute it for the uniform distribution
            // This makes this algorithm deterministic and may also provide some domain coverage advantages over a
            // psuedo-random sample.
            double value = -log ( halton ( sequence_pos,primes[i] ) );
            point_sum += value;
            point.push_back ( value );
          }
        for ( auto i = point.begin(); i != point.end(); ++i ) *i /= point_sum; // Normalize point to sum to 1
        func ( point );
        if ( point_dimension == 1 ) break; // no need to generate additional points; only one feasible point exists for 0-simplex
      }
  }

#endif

