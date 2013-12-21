/*=============================================================================
    Copyright (c) 2005 Robbie Vogt
	Copyright (c) 2012-2013 Richard Otis

	Based on code from the Boost mailing list: http://lists.boost.org/ublas/2005/07/0568.php

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Cholesky factorizations in the spirit of lu.hpp
// TODO: The full Cholesky factorization may not be needed; may need to write a simplified version to check for p.d. matrix

#ifndef BOOST_UBLAS_CHOLESKY_HPP
#define BOOST_UBLAS_CHOLESKY_HPP

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/triangular.hpp>

namespace boost { namespace numeric { namespace ublas {

  // Cholesky factorization
  template<class M>
  bool cholesky_factorize (M &m) {
    typedef M matrix_type;
    typedef typename M::size_type size_type;
    typedef typename M::value_type value_type;

    BOOST_UBLAS_CHECK (m.size1() == m.size2(), external_logic("Cholesky decomposition is only valid for a square, positive definite matrix."));

    size_type size = m.size1();
    vector<value_type> d(size);

    for (size_type i = 0; i < size; ++ i) {
      matrix_row<M> mri (row (m, i));
      for (size_type j = i; j < size; ++ j) {
        matrix_row<M> mrj (row (m, j));

        value_type elem = m(i,j) - inner_prod(project(mri,range(0,i)), project(mrj,range(0,i)));

        if (i == j) {
          if (elem <= 0.0) {
            // matrix after rounding errors is not positive definite
            return false;
          }
          else {
            d(i) = type_traits<value_type>::sqrt(elem);
          }
        }
        else {
          m(j,i) = elem / d(i);
        }
      }
    }
    // put the diagonal back in
    for (size_type i = 0; i < size; ++ i) {
      m(i,i) = d(i);
    }
    // decomposition succeeded
    return true;
  }

    // Cholesky substitution
  template<class M, class E>
  void cholesky_substitute (const M &m, vector_expression<E> &e) {
    typedef const M const_matrix_type;
    typedef vector<typename E::value_type> vector_type;

    inplace_solve (m, e, lower_tag ());
    inplace_solve (trans(m), e, upper_tag ());
  }
  template<class M, class E>
  void cholesky_substitute (const M &m, matrix_expression<E> &e) {
    typedef const M const_matrix_type;
    typedef matrix<typename E::value_type> matrix_type;

    inplace_solve (m, e, lower_tag ());
    inplace_solve (trans(m), e, upper_tag ());
  }
  template<class E, class M>
  void cholesky_substitute_left (vector_expression<E> &e, const M &m) {
    typedef const M const_matrix_type;
    typedef vector<typename E::value_type> vector_type;

    inplace_solve (trans(m), e, upper_tag ());
    inplace_solve (m, e, lower_tag ());
  }
  template<class E, class M>
  void cholesky_substitute_left (matrix_expression<E> &e, const M &m) {
    typedef const M const_matrix_type;
    typedef matrix<typename E::value_type> matrix_type;

    inplace_solve (trans(m), e, upper_tag ());
    inplace_solve (m, e, lower_tag ());
  }

  // Cholesky matrix inversion
  template<class M>
  void cholesky_invert (M &m)
  {
    typedef typename M::size_type size_type;
    typedef typename M::value_type value_type;

    size_type size = m.size1();

    // determine the inverse of the lower traingular matrix
    for (size_type i = 0; i < size; ++ i) {
      m(i,i) = 1 / m(i,i);

      for (size_type j = i+1; j < size; ++ j) {
        value_type elem(0);

        for (size_type k = i; k < j; ++ k) {
          elem -= m(j,k)*m(k,i);
        }
        m(j,i) = elem / m(j,j);
      }
    }

    // multiply the upper and lower inverses together
    m = prod(trans(triangular_adaptor<M,lower>(m)), triangular_adaptor<M,lower>(m));
  }

}}}

#endif
