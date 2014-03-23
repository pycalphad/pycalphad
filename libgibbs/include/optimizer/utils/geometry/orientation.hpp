/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Calculation of the orientation of a point with respect to a hyperplane

#ifndef INCLUDED_ORIENTATION
#define INCLUDED_ORIENTATION
#include "libgibbs/include/utils/determinant.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/assert.hpp>

/* The orientation is the signed volume of the simplex spanned by the hyperplane
 * and the candidate point, up to a constant factor depending on the dimension of
 * the space.
 * Reference: Computing in Euclidean Geometry, 2nd ed., 1995, edited by Ding-Zhu Du and Frank Hwang
 * 
 */
template <typename VectorType, typename DataType>
double orientation ( const boost::numeric::ublas::matrix<DataType> &hyperplane_points , 
                  const VectorType &candidate_point ) {
    using namespace boost::numeric::ublas;
    BOOST_ASSERT ( candidate_point.size() == hyperplane_points.size2() );
    matrix<DataType> orientation_test_matrix (hyperplane_points.size1()+1, hyperplane_points.size2()+1);
    matrix_row<DataType> last_row ( orientation_test_matrix, orientation_test_matrix.size1() );
    matrix_column<DataType> last_column ( orientation_test_matrix, orientation_test_matrix.size2() );
    // Fill the last column with 1's
    for (auto i = 0; i < orientation_test_matrix.size1(); ++i) {
        last_column ( i ) = 1;
    }
    // Fill the upper left part of the test matrix with the hyperplane points
    subrange ( orientation_test_matrix, 0,hyperplane_points.size1(), 0,hyperplane_points.size2() ) = hyperplane_points;
    // Fill the last row with the coordinates of the candidate point
    for ( auto i = candidate_point.cbegin(); i != candidate_point.cend(); ++i) {
        last_row ( std::distance( candidate_point.cbegin(), i ) ) = *i;
    }
    // Calculate and return the determinant of the orientation test matrix
    return determinant ( orientation_test_matrix );
}
#endif