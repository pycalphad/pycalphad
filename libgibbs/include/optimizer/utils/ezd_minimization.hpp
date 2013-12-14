/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// declaration for functions related to EZD global minimization
// Reference: Maria Emelianenko, Zi-Kui Liu, and Qiang Du.
// "A new algorithm for the automation of phase diagram calculation." Computational Materials Science 35.1 (2006): 61-74.

#ifndef INCLUDED_EZD_MINIMIZATION
#define INCLUDED_EZD_MINIMIZATION

#include <boost/assert.hpp>
#include <vector>

namespace Optimizer {

/* The purpose of this point class is to mimic the Boost Geometry Library point concept,
 * but allow the dimension of the point to be defined at run-time
 */
template
<
    typename CoordinateType,
    typename CoordinateSystem
>
class point
{
public:

    inline point()
    {}

    inline CoordinateType const& get(std::size_t const& K) const
    {
        BOOST_ASSERT(K <= dim());
        return m_values[K];
    }

    inline void set(std::size_t const& K, CoordinateType const& value)
    {
        if (K > dim()) {
        	// expand dimensionality of point
        	m_values.resize(K-1, CoordinateType(0));
        }
        m_values[K] = value;
    }

private:
    inline std::size_t dim() const { return m_values.size(); }
    std::vector<CoordinateType> m_values;
};

}

namespace boost { namespace geometry
{
namespace traits
{
template
<
    typename CoordinateType,
    typename CoordinateSystem
>
struct tag<model::point<CoordinateType, CoordinateSystem> >
{
    typedef point_tag type;
};

template
<
    typename CoordinateType,
    typename CoordinateSystem
>
struct coordinate_type<model::point<CoordinateType, CoordinateSystem> >
{
    typedef CoordinateType type;
};

template
<
    typename CoordinateType,
    typename CoordinateSystem
>
struct coordinate_system<model::point<CoordinateType, CoordinateSystem> >
{
    typedef CoordinateSystem type;
};

template
<
    typename CoordinateType,
    typename CoordinateSystem
>
struct dimension<model::point<CoordinateType, CoordinateSystem> >
    : boost::mpl::int_<DimensionCount>
{};

template
<
    typename CoordinateType,
    std::size_t DimensionCount,
    typename CoordinateSystem,
    std::size_t Dimension
>
struct access<model::point<CoordinateType, DimensionCount, CoordinateSystem>, Dimension>
{
    static inline CoordinateType get(
        model::point<CoordinateType, DimensionCount, CoordinateSystem> const& p)
    {
        return p.template get<Dimension>();
    }

    static inline void set(
        model::point<CoordinateType, DimensionCount, CoordinateSystem>& p,
        CoordinateType const& value)
    {
        p.template set<Dimension>(value);
    }
};

#endif
