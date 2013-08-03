/*=============================================================================
    Copyright (c) 2012-2013 "Loki Astari" (StackExchange)
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Adapted from http://codereview.stackexchange.com/questions/14309/conversion-between-enum-and-string-in-c-class-header

#ifndef INCLUDED_ENUM_HANDLING
#define INCLUDED_ENUM_HANDLING

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

// This is the type that will hold all the strings.
// Each enumerate type will declare its own specialization.
// Any enum that does not have a specialization will generate a compiler error
// indicating that there is no definition of this variable (as there should be
// be no definition of a generic version).
template<typename T>
struct enumStrings
{
    static char const* data[];
};

// This is a utility type.
// Created automatically. Should not be used directly.
template<typename T>
struct enumRefHolder
{
    T& enumVal;
    enumRefHolder(T& enumVal): enumVal(enumVal) {}
};
template<typename T>
struct enumConstRefHolder
{
    T const& enumVal;
    enumConstRefHolder(T const& enumVal): enumVal(enumVal) {}
};

// The next two functions do the actual work of reading/writing an
// enum as a string.
template<typename T>
std::ostream& operator<<(std::ostream& str, enumConstRefHolder<T> const& data)
{
   return str << enumStrings<T>::data[static_cast<int>(data.enumVal)];
}

template<typename T>
std::istream& operator>>(std::istream& str, enumRefHolder<T> const& data)
{
    std::string value;
    str >> value;

    // These two can be made easier to read in C++11
    // using std::begin() and std::end()
    //
    static auto begin  = std::begin(enumStrings<T>::data);
    static auto end    = std::end(enumStrings<T>::data);

    auto find   = std::find(begin, end, value);
    if (find != end)
    {
        data.enumVal = static_cast<T>(std::distance(begin, find));
    }
    return str;
}


// This is the public interface:
// use the ability of function to deduce their template type without
// being explicitly told to create the correct type of enumRefHolder<T>
template<typename T>
enumConstRefHolder<T>  enumToString(T const& e) {return enumConstRefHolder<T>(e);}

template<typename T>
enumRefHolder<T>       enumFromString(T& e)     {return enumRefHolder<T>(e);}

#endif
