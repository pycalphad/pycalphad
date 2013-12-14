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
#include <boost/geometry/index/rtree.hpp>


namespace Optimizer {

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

void LocateMinima(std::unique_ptr<CompositionSet> const &phase, const int depth) {
	typedef bg::model::dynamic_point<float, bg::cs::cartesian> point;
	typedef bg::model::dynamic_box<point> box;
	typedef std::pair<box, double> value;

}

}
