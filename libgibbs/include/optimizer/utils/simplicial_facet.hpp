/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Simplicial facet of a convex hull

#ifndef INCLUDED_SIMPLICIAL_FACET
#define INCLUDED_SIMPLICIAL_FACET

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <vector>

namespace Optimizer { namespace details {
template <typename CoordinateType = double>
struct SimplicialFacet {
    typedef std::vector<CoordinateType> PointType;
    typedef boost::numeric::ublas::matrix<CoordinateType> MatrixType;
    CoordinateType area;
    PointType normal;
    std::vector<std::size_t> vertices;
    /* basis_matrix is the inverse of the matrix of independent vertices.
     * The purpose is to be able to quickly calculate whether points
     * are inside the facet.
     * Prior to inversion:
     * Each row is the independent components of the vertex.
     * Each column is a composition coordinate, with the last
     * column as all 1's.
     */
    MatrixType basis_matrix;
};
} // namespace details
} // namespace Optimizer
#endif