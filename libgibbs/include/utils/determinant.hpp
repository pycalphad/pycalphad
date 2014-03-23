/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Calculation of the determinant of a matrix

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/assert.hpp>

template <typename T>
T determinant ( const boost::numeric::ublas::matrix<T> &input ) {
    using namespace boost::numeric::ublas;
    matrix<T> A ( input ); // copy input matrix to A
    T det = 1;
    permutation_matrix<std::size_t,T> pm ( A.size1() );
    auto error = lu_factorize ( A, pm ); // calculate LU factorization of A; save in A
    BOOST_ASSERT ( error == 0 );
    for (auto i = 0; i < A.size1(); ++i) {
        det *= (pm(i) == i ? 1 : -1) * A ( i,i );
    }
    return det;
}