/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/
 
 #ifndef INCLUDED_INVERT_MATRIX
 #define INCLUDED_INVERT_MATRIX

 #include <boost/numeric/ublas/vector.hpp>
 #include <boost/numeric/ublas/vector_proxy.hpp>
 #include <boost/numeric/ublas/matrix.hpp>
 #include <boost/numeric/ublas/triangular.hpp>
 #include <boost/numeric/ublas/lu.hpp>
 #include <boost/numeric/ublas/io.hpp>

 /* Matrix inversion routine
  * Uses lu_factorize and lu_substitute to invert a matrix 
  * Reference: Numerical Recipies in C, 2nd ed., by Press, Teukolsky, Vetterling & Flannery.
  */
 template<class T>
 bool InvertMatrix (const boost::numeric::ublas::matrix<T>& input, 
                    boost::numeric::ublas::matrix<T>& inverse) {
     using namespace boost::numeric::ublas;
     typedef permutation_matrix<std::size_t> pmatrix;
     // create a working copy of the input
     matrix<T> A(input);
     // create a permutation matrix for the LU-factorization
     pmatrix pm(A.size1());
     // perform LU-factorization
     int res = lu_factorize(A,pm);
     if( res != 0 ) return false;
     // create identity matrix of "inverse"
     inverse.assign(identity_matrix<T>(A.size1()));
     // backsubstitute to get the inverse
     lu_substitute(A, pm, inverse);
     return true;
 }
 
 #endif