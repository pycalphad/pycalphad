/*=============================================================================
    Copyright (c) 2007-2012 Barend Gehrels, Amsterdam, the Netherlands.
    Copyright (c) 2008-2012 Bruno Lalande, Paris, France.
    Copyright (c) 2009-2012 Mateusz Loskot, London, UK.
	Copyright (c) 2012-2013 Richard Otis

	Based on code from the Boost Geometry Library

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#ifndef INCLUDED_DYNAMIC_BOX
#define INCLUDED_DYNAMIC_BOX

#include <boost/geometry/core/access.hpp>


namespace boost { namespace geometry
{

namespace model
{


/*!
    \brief Class dynamic_box: defines a box made of two describing points
    \ingroup geometries
    \details Box is always described by a min_corner() and a max_corner() point. If another
        rectangle is used, use linear_ring or polygon.
    \note Boxes are for selections and for calculating the envelope of geometries. Not all algorithms
    are implemented for box. Boxes are also used in Spatial Indexes.
    \tparam Point point type. The box takes a point type as template parameter.
    The point type can be any point type.
    It can be 2D but can also be 3D or more dimensional.
    The box can also take a latlong point type as template parameter.
 */

template<typename Point>
class dynamic_box
{

public:

    inline dynamic_box() {}

    /*!
        \brief Constructor taking the minimum corner point and the maximum corner point
    */
    inline dynamic_box(Point const& min_corner, Point const& max_corner)
    {
    	m_min_corner = min_corner;
    	m_max_corner = max_corner;
    }

    inline Point const& min_corner() const { return m_min_corner; }
    inline Point const& max_corner() const { return m_max_corner; }

    inline Point& min_corner() { return m_min_corner; }
    inline Point& max_corner() { return m_max_corner; }

private:

    Point m_min_corner;
    Point m_max_corner;
};


} // namespace model


// Traits specializations for box above
#ifndef DOXYGEN_NO_TRAITS_SPECIALIZATIONS
namespace traits
{

template <typename Point>
struct tag<model::dynamic_box<Point> >
{
    typedef box_tag type;
};

template <typename Point>
struct point_type<model::dynamic_box<Point> >
{
    typedef Point type;
};

template <typename Point>
struct dynamic_indexed_access<model::dynamic_box<Point> >
{
	typedef typename geometry::coordinate_type<Point>::type coordinate_type;

	static inline coordinate_type get(std::size_t const& Index, model::dynamic_box<Point> const& b)
	{
		if (Index == min_corner) {
			return geometry::get(b.min_corner());
		}
		else if (Index == max_corner) {
			return geometry::get(b.max_corner());
		}
	}

	static inline void set(model::dynamic_box<Point>& b, std::size_t const& Index, coordinate_type const& value)
    {
    	if (Index == min_corner) {
    		geometry::set(b.min_corner(), Index, value);
    	}
    	else if (Index == max_corner) {
    		geometry::set(b.max_corner(), Index, value);
    	}
    }
};

} // namespace traits
#endif // DOXYGEN_NO_TRAITS_SPECIALIZATIONS

}} // namespace boost::geometry

#endif // BOOST_GEOMETRY_GEOMETRIES_BOX_HPP
