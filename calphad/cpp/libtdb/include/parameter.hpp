/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// definitions for Parameter object

#ifndef INCLUDED_PARAMETER
#define INCLUDED_PARAMETER

#include <boost/spirit/home/support/utree.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/fusion/include/adapt_struct.hpp>

struct Parameter {
    std::string phase; // name of the phase to which the parameter applies
    std::string suffix; // special indicator after underscore character: B2, A2, L12, LAVES, etc.
    std::string type; // parameter type: G, L, TC, BMAGN, etc.
    std::vector<std::vector<std::string>> constituent_array; // sublattice conditions that must be met for parameter to apply
    int degree;                     // degree of Redlich-Kister term (if applicable)
    boost::spirit::utree ast; // abstract syntax tree associated with parameter (arithmetic expression with limits)
    //std::string data_ref; // scientific reference for the parameter
    
    std::string phasename() const {
        if (!suffix.empty()) {
            std::string str = phase + "_" + suffix;
            return str;
        }
        else return phase;
    }
    int wildcount() const {
        const auto array_begin = constituent_array.begin();
        const auto array_end = constituent_array.end();
        int wilds;
        for (auto j = array_begin; j != array_end; ++j) {
            if ((*j).size() == 1 && (*j)[0] == "*") {
                ++wilds;
            }
        }
        return wilds;
    }
};
BOOST_FUSION_ADAPT_STRUCT
(
    Parameter,
 (std::string, type)
 (std::string, phase)
 (std::string, suffix)
 (std::vector<std::vector<std::string>>, constituent_array)
 (int, degree)
 (spirit::utree, ast)
)
typedef std::vector<Parameter> Parameters;

struct type_index {};
struct phase_index {};

typedef boost::multi_index_container<
Parameter,
boost::multi_index::indexed_by<
boost::multi_index::ordered_non_unique<
boost::multi_index::tag<phase_index>,
BOOST_MULTI_INDEX_CONST_MEM_FUN(Parameter,std::string,phasename)
>,
boost::multi_index::ordered_non_unique<
boost::multi_index::tag<type_index>,
BOOST_MULTI_INDEX_MEMBER(Parameter,std::string,type)
>
>
> parameter_set;

typedef boost::multi_index_container<
const Parameter*,
boost::multi_index::indexed_by<
boost::multi_index::ordered_non_unique<
boost::multi_index::tag<phase_index>,
BOOST_MULTI_INDEX_MEMBER(Parameter,const std::string,phase)
>,
boost::multi_index::ordered_non_unique<
boost::multi_index::tag<type_index>,
BOOST_MULTI_INDEX_MEMBER(Parameter,const std::string,type)
>
>
> parameter_set_view;


#endif