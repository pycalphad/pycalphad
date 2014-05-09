/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// utility function for applying functors to unique pairs of elements in a container

#ifndef INCLUDED_FOR_EACH_PAIR
#define INCLUDED_FOR_EACH_PAIR

template <typename Iterator, typename Functor> void for_each_pair (Iterator begin, Iterator end, Functor f) {
    for (auto  i = begin; i != end; ++i) {
        for (auto j = i; ++j != end; ) {
            f (i,j);
        }
    }
}

#endif