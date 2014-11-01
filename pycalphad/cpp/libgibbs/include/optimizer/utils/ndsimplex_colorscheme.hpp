/*=============================================================================
 Copyright (c) *2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// Color scheme generation for simplex subdivision

#ifndef INCLUDED_NDSIMPLEX_COLORSCHEME
#define INCLUDED_NDSIMPLEX_COLORSCHEME

#include <boost/numeric/ublas/matrix.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
namespace Optimizer
{
	namespace details
	{
		typedef boost::numeric::ublas::matrix<std::size_t> ColorMatrixType;
		std::vector<std::size_t> decimal_to_base ( std::size_t decimal, std::size_t base, std::size_t digits )
		{
			/*% converts the decimal number "decimal" to y in base "base" with "digits" digits:
			 * % y(1)+y(2)*b+y(3)*b^2+...+y(n)*b^(n-1) = x
			 */
			std::vector<std::size_t> y;
			y.reserve ( digits );
			std::size_t x = decimal;
			for ( auto i = 0; i < digits; ++i )
			{
				std::size_t d = x / base; // float->unsigned conversion will round towards zero as desired
				y.push_back ( x - d*base );
				x = d;
			}
			return y;
		}
	}
	
	/*
	 * This algorithm is an easy way to construct a color scheme for the subdivision.
	 * It will generate the k^d color schemes for d-dimensional simplex subdivision.
	 * Reference: Goncalves, Palhares, Takahashi, and Mesquita, 2006.
	 *   "Algorithm 860: SimpleS-An Extension of Freudenthal's Simplex Subdivision"
	 */
	std::vector<details::ColorMatrixType> generate_color_schemes ( std::size_t k, std::size_t d )
	{
		BOOST_ASSERT ( k > 0 );
		BOOST_ASSERT ( d > 0 );
		const std::size_t maxcolors = std::pow ( k,d );
		std::vector<details::ColorMatrixType> colors ( maxcolors, details::ColorMatrixType ( k,d+1 ) );
		
		for ( auto n = 0; n < maxcolors; ++n )
		{
			const std::vector<std::size_t> x = details::decimal_to_base ( n, k, d );
			std::size_t color = 0;
			for ( auto i = 0; i < k; ++i )
			{
				( colors[n] ) ( i,0 ) = color;
				for ( auto j = 1; j < d+1; ++j )
				{
					if ( x[d-j] == i ) color = color + 1;
					( colors[n] ) ( i,j ) = color;
				}
			}
			
		}
		return colors;
	}
	
}
#endif