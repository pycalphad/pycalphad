/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// implementation of Boost Geometry Library point concept with run-time dimensionality

#include <boost/geometry/core/access.hpp>
#include <boost/geometry/core/coordinate_type.hpp>
#include <boost/geometry/core/coordinate_system.hpp>
#include <boost/geometry/core/tag.hpp>
#include <vector>

namespace boost { namespace geometry {

namespace model {

/* The purpose of this point class is to mimic the Boost Geometry Library point concept,
 * but allow the dimension of the point to be defined at run-time
 */
template
<
    typename CoordinateType,
    typename CoordinateSystem
>
class dynamic_point
{
public:

    inline dynamic_point()
    {}

    inline CoordinateType const& get(std::size_t const& K) const
    {
        if (K >= dim()) {
        	return CoordinateType(0);
        }
        else return m_values[K];
    }

    inline void set(std::size_t const& K, CoordinateType const& value)
    {
        if (K >= dim()) {
        	// expand dimensionality of point
        	m_values.resize(K-1, CoordinateType(0));
        }
        m_values[K] = value;
    }
    inline std::size_t dim() const { return m_values.size(); }
private:
    std::vector<CoordinateType> m_values;
};

} // namespace model

namespace traits
{
template
<
    typename CoordinateType,
    typename CoordinateSystem
>
struct tag<Optimizer::point<CoordinateType, CoordinateSystem> >
{
    typedef point_tag type;
};

template
<
    typename CoordinateType,
    typename CoordinateSystem
>
struct coordinate_type<Optimizer::point<CoordinateType, CoordinateSystem> >
{
    typedef CoordinateType type;
};

template
<
    typename CoordinateType,
    typename CoordinateSystem
>
struct coordinate_system<Optimizer::point<CoordinateType, CoordinateSystem> >
{
    typedef CoordinateSystem type;
};

template
<
    typename CoordinateType,
    typename CoordinateSystem
>
struct access<Optimizer::point<CoordinateType, CoordinateSystem> >
{
    static inline CoordinateType get(
        std::size_t const& K, Optimizer::point<CoordinateType, CoordinateSystem> const& p)
    {
        return p.get(K);
    }

    static inline void set(
    	std::size_t const& K,
        Optimizer::point<CoordinateType, CoordinateSystem>& p,
        CoordinateType const& value)
    {
        p.set(K, value);
    }
};

} // namespace traits
} // namespace geometry
} // namespace boost
