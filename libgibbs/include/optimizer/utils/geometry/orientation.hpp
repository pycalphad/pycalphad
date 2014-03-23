/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Calculation of the orientation of a point with respect to a hyperplane

#ifndef INCLUDED_ORIENTATION
#define INCLUDED_ORIENTATION
#include "libgibbs/include/optimizer/utils/ndsimplex.hpp"

/* The orientation is the signed volume of the simplex spanned by the hyperplane
 * and the candidate point, up to a constant factor depending on the dimension of
 * the space.
 * Reference: Computing in Euclidean Geometry, 2nd ed., 1995, edited by Ding-Zhu Du and Frank Hwang
 * Here the vertices of an NDSimplex are used to define the hyperplane.
 */
template <typename T>
int orientation ( const NDSimplex &simplex , const T &candidate_point ) {
    return 0;
}
#endif