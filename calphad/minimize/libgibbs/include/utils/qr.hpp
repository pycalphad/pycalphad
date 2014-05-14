/*=============================================================================
    Copyright (c) 2010-2013 Karl Rupp (rupp@iue.tuwien.ac.at)
	Copyright (c) 2012-2013 Richard Otis

	Based on code from ViennaCL, the Vienna Computing Library
	Karl Rupp gave permission to relicense for Boost uBLAS:
	http://boost.2283326.n4.nabble.com/Matrix-decompositions-tp4647312p4647391.html
	"Feel free to copy&paste and relicense as needed, I agree to whatever is
necessary to integrate into uBLAS if of interest." -- Karl Rupp, May 16, 2013

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/


// Provides a QR factorization using a block-based approach.

#ifndef INCLUDED_QR_HPP
#define INCLUDED_QR_HPP

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <math.h>
#include <cmath>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>

template <typename MatrixType, typename VectorType>
typename MatrixType::value_type setup_householder_vector_ublas(
		MatrixType const & A, VectorType & v, MatrixType & matrix_1x1, std::size_t j)
{
	using boost::numeric::ublas::range;
	using boost::numeric::ublas::project;
	using boost::numeric::ublas::trans;

	typedef typename MatrixType::value_type   ScalarType;

	//compute norm of column below diagonal:
	matrix_1x1 = boost::numeric::ublas::prod( trans(project(A, range(j+1, A.size1()), range(j, j+1))),
			project(A, range(j+1, A.size1()), range(j, j+1))
	);
	ScalarType sigma = matrix_1x1(0,0);
	ScalarType beta = 0;
	ScalarType A_jj = A(j,j);

	assert( sigma >= 0.0  && bool("sigma must be non-negative!"));

	//get v from A:
	v(j,0) = 1.0;
	project(v, range(j+1, A.size1()), range(0,1)) = project(A, range(j+1, A.size1()), range(j,j+1));

	if (sigma == 0)
		return 0;
	else
	{
		ScalarType mu = std::sqrt(sigma + A_jj*A_jj);

		ScalarType v1 = (A_jj <= 0) ? (A_jj - mu) : (-sigma / (A_jj + mu));
		beta = static_cast<ScalarType>(2.0) * v1 * v1 / (sigma + v1 * v1);

		//divide v by its diagonal element v[j]
		                                     project(v, range(j+1, A.size1()), range(0,1)) /= v1;
	}

	return beta;
}


// Apply (I - beta v v^T) to the k-th column of A, where v is the reflector starting at j-th row/column
template <typename MatrixType, typename VectorType, typename ScalarType>
void householder_reflect(MatrixType & A, VectorType & v, ScalarType beta, std::size_t j, std::size_t k)
{
	ScalarType v_in_col = A(j,k);
	for (auto i=j+1; i<A.size1(); ++i)
		v_in_col += v[i] * A(i,k);

	//assert(v[j] == 1.0);

	for (auto i=j; i<A.size1(); ++i)
		A(i,k) -= beta * v_in_col * v[i];
}

template <typename MatrixType, typename VectorType, typename ScalarType>
void householder_reflect_ublas(
		MatrixType & A, VectorType & v, MatrixType & matrix_1x1, ScalarType beta, std::size_t j, std::size_t k)
{
	using boost::numeric::ublas::range;
	using boost::numeric::ublas::project;
	using boost::numeric::ublas::trans;

	ScalarType v_in_col = A(j,k);
	matrix_1x1 = boost::numeric::ublas::prod(trans(project(v, range(j+1, A.size1()), range(0, 1))),
			project(A, range(j+1, A.size1()), range(k,k+1)));
	v_in_col += matrix_1x1(0,0);

	project(A, range(j, A.size1()), range(k, k+1)) -= (beta * v_in_col) * project(v, range(j, A.size1()), range(0, 1));
}

// Apply (I - beta v v^T) to A, where v is the reflector starting at j-th row/column
template <typename MatrixType, typename VectorType, typename ScalarType>
void householder_reflect(MatrixType & A, VectorType & v, ScalarType beta, std::size_t j)
{
	std::size_t column_end = A.size2();

	for (std::size_t k=j; k<column_end; ++k) //over columns
		householder_reflect(A, v, beta, j, k);
}


template <typename MatrixType, typename VectorType>
void write_householder_to_A(MatrixType & A, VectorType const & v, std::size_t j)
{
	for (std::size_t i=j+1; i<A.size1(); ++i)
		A(i,j) = v[i];
}

template <typename MatrixType, typename VectorType>
void write_householder_to_A_ublas(MatrixType & A, VectorType const & v, std::size_t j)
{
	using boost::numeric::ublas::range;
	using boost::numeric::ublas::project;

	//VectorType temp = project(v, range(j+1, A.size1()));
	project( A, range(j+1, A.size1()), range(j, j+1) ) = project(v, range(j+1, A.size1()), range(0, 1) );;
}


/** @brief Implementation of inplace-QR factorization for a general Boost.uBLAS compatible matrix A
 *
 * @param A            A dense compatible to Boost.uBLAS
 * @param block_size   The block size to be used. The number of columns of A must be a multiple of block_size
 */
template<typename MatrixType>
std::vector<typename MatrixType::value_type> inplace_qr_ublas(MatrixType & A, std::size_t block_size = 32)
{
	typedef typename MatrixType::value_type   ScalarType;
	typedef boost::numeric::ublas::matrix_range<MatrixType>  MatrixRange;

	using boost::numeric::ublas::range;
	using boost::numeric::ublas::project;

	std::vector<ScalarType> betas(A.size2());
	MatrixType v(A.size1(), 1);
	MatrixType matrix_1x1(1,1);

	MatrixType Y(A.size1(), block_size); Y.clear(); Y.resize(A.size1(), block_size);
	MatrixType W(A.size1(), block_size); W.clear(); W.resize(A.size1(), block_size);

	//run over A in a block-wise manner:
	for (std::size_t j = 0; j < std::min(A.size1(), A.size2()); j += block_size)
	{
		std::size_t effective_block_size = std::min(std::min(A.size1(), A.size2()), j+block_size) - j;

		//determine Householder vectors:
		for (std::size_t k = 0; k < effective_block_size; ++k)
		{
			betas[j+k] = setup_householder_vector_ublas(A, v, matrix_1x1, j+k);

			for (std::size_t l = k; l < effective_block_size; ++l)
				householder_reflect_ublas(A, v, matrix_1x1, betas[j+k], j+k, j+l);

			write_householder_to_A_ublas(A, v, j+k);
		}

		//
		// Setup Y:
		//
		Y.clear();  Y.resize(A.size1(), block_size);
		for (std::size_t k = 0; k < effective_block_size; ++k)
		{
			//write Householder to Y:
			Y(j+k,k) = 1.0;
			project(Y, range(j+k+1, A.size1()), range(k, k+1)) = project(A, range(j+k+1, A.size1()), range(j+k, j+k+1));
		}

		//
		// Setup W:
		//

		//first vector:
		W.clear();  W.resize(A.size1(), block_size);
		W(j, 0) = -betas[j];
		project(W, range(j+1, A.size1()), range(0, 1)) = -betas[j] * project(A, range(j+1, A.size1()), range(j, j+1));


		//k-th column of W is given by -beta * (Id + W*Y^T) v_k, where W and Y have k-1 columns
		for (std::size_t k = 1; k < effective_block_size; ++k)
		{
			MatrixRange Y_old = project(Y, range(j, A.size1()), range(0, k));
			MatrixRange v_k   = project(Y, range(j, A.size1()), range(k, k+1));
			MatrixRange W_old = project(W, range(j, A.size1()), range(0, k));
			MatrixRange z     = project(W, range(j, A.size1()), range(k, k+1));

			MatrixType YT_prod_v = boost::numeric::ublas::prod(boost::numeric::ublas::trans(Y_old), v_k);
			z = - betas[j+k] * (v_k + prod(W_old, YT_prod_v));
		}

		//
		//apply (I+WY^T)^T = I + Y W^T to the remaining columns of A:
		//

		if (A.size2() - j - effective_block_size > 0)
		{

			MatrixRange A_part(A, range(j, A.size1()), range(j+effective_block_size, A.size2()));
			MatrixRange W_part(W, range(j, A.size1()), range(0, effective_block_size));
			MatrixType temp = boost::numeric::ublas::prod(trans(W_part), A_part);

			A_part += prod(project(Y, range(j, A.size1()), range(0, Y.size2())),
					temp);
		}
	}

	return betas;
}





//takes an inplace QR matrix A and generates Q and R explicitly
template <typename MatrixType, typename VectorType>
void recoverQ(MatrixType const & A, VectorType const & betas, MatrixType & Q, MatrixType & R)
{
	typedef typename MatrixType::value_type   ScalarType;

	std::vector<ScalarType> v(A.size1());

	Q.clear();
	R.clear();

	//
	// Recover R from upper-triangular part of A:
	//
	std::size_t i_max = std::min(R.size1(), R.size2());
	for (std::size_t i=0; i<i_max; ++i)
		for (std::size_t j=i; j<R.size2(); ++j)
			R(i,j) = A(i,j);

	//
	// Recover Q by applying all the Householder reflectors to the identity matrix:
	//
	for (std::size_t i=0; i<Q.size1(); ++i)
		Q(i,i) = 1.0;

	std::size_t j_max = std::min(A.size1(), A.size2());
	for (std::size_t j=0; j<j_max; ++j)
	{
		std::size_t col_index = j_max - j - 1;
		v[col_index] = 1.0;
		for (std::size_t i=col_index+1; i<A.size1(); ++i)
			v[i] = A(i, col_index);

		if (betas[col_index] != 0)
			householder_reflect(Q, v, betas[col_index], col_index);
	}
}

/** @brief Overload of inplace-QR factorization for a general Boost.uBLAS compatible matrix A
 *
 * @param A            A dense compatible to Boost.uBLAS
 * @param block_size   The block size to be used.
 */
template<typename MatrixType>
std::vector<typename MatrixType::value_type> inplace_qr(MatrixType & A, std::size_t block_size = 16)
{
	return inplace_qr_ublas(A, block_size);
}


#endif
