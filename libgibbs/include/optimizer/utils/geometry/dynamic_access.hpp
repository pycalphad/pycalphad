/*=============================================================================
    Copyright (c) 2008-2012 Bruno Lalande, Paris, France.
    Copyright (c) 2008-2012 Barend Gehrels, Amsterdam, the Netherlands.
    Copyright (c) 2009-2012 Mateusz Loskot, London, UK.
	Copyright (c) 2012-2013 Richard Otis

	Based on code from the Boost Geometry Library

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Implementation of access for run-time geometry

#ifndef INCLUDED_DYNAMIC_ACCESS
#define INCLUDED_DYNAMIC_ACCESS


#include <boost/geometry/core/access.hpp>


namespace boost { namespace geometry
{

namespace traits
{

/*!
\brief Traits class which gives access (get,set) to points.
\ingroup traits
\par Geometries:
///     @li point
\par Specializations should provide
///     @li static inline T get(G const&)
///     @li static inline void set(G&, T const&)
\tparam Geometry geometry-type
*/
template <typename Geometry, typename Enable = void>
struct dynamic_access
{
   BOOST_MPL_ASSERT_MSG
        (
            false, NOT_IMPLEMENTED_FOR_THIS_POINT_TYPE, (types<Geometry>)
        );
};


/*!
\brief Traits class defining "get" and "set" to get
    and set point coordinate values
\tparam Geometry geometry (box, segment)
\par Geometries:
    - box
    - segment
\par Specializations should provide:
    - static inline T get(std::size_t const&, G const&)
    - static inline void set(G&, std::size_t const&, T const&)
\ingroup traits
*/
template <typename Geometry>
struct dynamic_indexed_access {};


} // namespace traits

#ifndef DOXYGEN_NO_DETAIL
namespace detail
{

template
<
    typename Geometry,
    typename CoordinateType
>
struct dynamic_indexed_access_non_pointer
{
    static inline CoordinateType get(std::size_t const& Index, Geometry const& geometry)
    {
        return traits::dynamic_indexed_access<Geometry>::get(Index, geometry);
    }
    static inline void set(Geometry& b, std::size_t const &Index, CoordinateType const& value)
    {
        traits::dynamic_indexed_access<Geometry>::set(b, Index, value);
    }
};

template
<
    typename Geometry,
    typename CoordinateType
>
struct dynamic_indexed_access_pointer
{
    static inline CoordinateType get(std::size_t const &Index, Geometry const* geometry)
    {
        return traits::dynamic_indexed_access<typename boost::remove_pointer<Geometry>::type>::get(Index, *geometry);
    }
    static inline void set(Geometry* geometry, std::size_t const &Index, CoordinateType const& value)
    {
        traits::dynamic_indexed_access<typename boost::remove_pointer<Geometry>::type>::set(*geometry, Index, value);
    }
};


} // namespace detail
#endif // DOXYGEN_NO_DETAIL


#ifndef DOXYGEN_NO_DISPATCH
namespace core_dispatch
{

template
<
    typename Tag,
    typename Geometry,
    typename CoordinateType,
    typename IsPointer
>
struct dynamic_access
{
    //static inline T get(G const&) {}
    //static inline void set(G& g, T const& value) {}
};

template
<
    typename Tag,
    typename Geometry,
    typename CoordinateType,
    typename IsPointer
>
struct dynamic_indexed_access
{
    //static inline T get(G const&) {}
    //static inline void set(G& g, T const& value) {}
};

template <typename Point, typename CoordinateType>
struct dynamic_access<point_tag, Point, CoordinateType, boost::false_type>
{
    static inline CoordinateType get(std::size_t const& Index, Point const& point)
    {
        return traits::dynamic_access<Point>::get(Index, point);
    }
    static inline void set(Point& p, std::size_t const& Index, CoordinateType const& value)
    {
        traits::dynamic_access<Point>::set(p, Index, value);
    }
};

template <typename Point, typename CoordinateType>
struct dynamic_access<point_tag, Point, CoordinateType, boost::true_type>
{
    static inline CoordinateType get(std::size_t const& Index, Point const* point)
    {
        return traits::dynamic_access<typename boost::remove_pointer<Point>::type>::get(Index, *point);
    }
    static inline void set(Point* p, std::size_t const& Index, CoordinateType const& value)
    {
        traits::dynamic_access<typename boost::remove_pointer<Point>::type>::set(*p, Index, value);
    }
};


template
<
    typename Box,
    typename CoordinateType
>
struct dynamic_indexed_access<box_tag, Box, CoordinateType, boost::false_type>
    : detail::dynamic_indexed_access_non_pointer<Box, CoordinateType>
{};

template
<
    typename Box,
    typename CoordinateType
>
struct dynamic_indexed_access<box_tag, Box, CoordinateType, boost::true_type>
    : detail::dynamic_indexed_access_pointer<Box, CoordinateType>
{};


template
<
    typename Segment,
    typename CoordinateType
>
struct dynamic_indexed_access<segment_tag, Segment, CoordinateType, boost::false_type>
    : detail::dynamic_indexed_access_non_pointer<Segment, CoordinateType>
{};


template
<
    typename Segment,
    typename CoordinateType
>
struct dynamic_indexed_access<segment_tag, Segment, CoordinateType, boost::true_type>
    : detail::dynamic_indexed_access_pointer<Segment, CoordinateType>
{};

} // namespace core_dispatch
#endif // DOXYGEN_NO_DISPATCH


template <typename Geometry>
inline typename coordinate_type<Geometry>::type get(std::size_t const& Index, Geometry const& geometry
#ifndef DOXYGEN_SHOULD_SKIP_THIS
        , detail::signature_getset_dimension* dummy = 0
#endif
        )
{
    boost::ignore_unused_variable_warning(dummy);

    typedef core_dispatch::dynamic_access
        <
            typename tag<Geometry>::type,
            typename geometry::util::bare_type<Geometry>::type,
            typename coordinate_type<Geometry>::type,
            typename boost::is_pointer<Geometry>::type
        > coord_access_type;

    return coord_access_type::get(Index, geometry);
}


template <typename Geometry>
inline void set(Geometry& geometry,
		std::size_t const& Index,
        typename coordinate_type<Geometry>::type const& value
#ifndef DOXYGEN_SHOULD_SKIP_THIS
        , detail::signature_getset_dimension* dummy = 0
#endif
        )
{
    boost::ignore_unused_variable_warning(dummy);

    typedef core_dispatch::dynamic_access
        <
            typename tag<Geometry>::type,
            typename geometry::util::bare_type<Geometry>::type,
            typename coordinate_type<Geometry>::type,
            typename boost::is_pointer<Geometry>::type
        > coord_access_type;

    coord_access_type::set(geometry, Index, value);
}


template <typename Geometry>
inline typename coordinate_type<Geometry>::type get(Geometry const& geometry,
		std::size_t const& Index
#ifndef DOXYGEN_SHOULD_SKIP_THIS
        , detail::signature_getset_index_dimension* dummy = 0
#endif
        )
{
    boost::ignore_unused_variable_warning(dummy);

    typedef core_dispatch::dynamic_indexed_access
        <
            typename tag<Geometry>::type,
            typename geometry::util::bare_type<Geometry>::type,
            typename coordinate_type<Geometry>::type,
            typename boost::is_pointer<Geometry>::type
        > coord_access_type;

    return coord_access_type::get(Index, geometry);
}

template <typename Geometry>
inline void set(Geometry& geometry,
		std::size_t const& Index,
        typename coordinate_type<Geometry>::type const& value
#ifndef DOXYGEN_SHOULD_SKIP_THIS
        , detail::signature_getset_index_dimension* dummy = 0
#endif
        )
{
    boost::ignore_unused_variable_warning(dummy);

    typedef core_dispatch::dynamic_indexed_access
        <
            typename tag<Geometry>::type,
            typename geometry::util::bare_type<Geometry>::type,
            typename coordinate_type<Geometry>::type,
            typename boost::is_pointer<Geometry>::type
        > coord_access_type;

    coord_access_type::set(geometry, Index, value);
}

}} // namespace boost::geometry

#endif
