/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// subroutines for EZD global minimization
// Reference: Maria Emelianenko, Zi-Kui Liu, and Qiang Du.
// "A new algorithm for the automation of phase diagram calculation." Computational Materials Science 35.1 (2006): 61-74.

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/optimizer/utils/ezd_minimization.hpp"
#include "libgibbs/include/compositionset.hpp"
#include "libgibbs/include/optimizer/halton.hpp"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <vector>
#include <tuple>

namespace Optimizer {

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

// MAX_DEGREES_FREEDOM is a compile-time constant due to the design of the Boost Geometry Library
constexpr const int MAX_DEGREES_FREEDOM = 20; // Maximum dimensionality of global minimization space
constexpr const int MAX_REFINEMENTS = 5;
typedef bg::model::point<float, MAX_DEGREES_FREEDOM, bg::cs::cartesian> Point;
typedef bg::model::box<Point> Box;
typedef std::pair<Box, double> Value;

void LocateMinima(const Region &domain, std::unique_ptr<CompositionSet> const &phase, const int depth) {
	bgi::rtree< Value, bgi::quadratic<16> > rtree;

    // create some values
    for ( double i = 0 ; i < 10 ; ++i )
    {
        // create a box
        Box b(Point(i, i), point(i + 0.5f, i + 0.5f));
        // insert new value
        rtree.insert(std::make_pair(b, i));
    }
}

}
