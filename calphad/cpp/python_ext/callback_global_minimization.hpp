/*=============================================================================
 Copyright (c) 2012-2014 Richard Otis
 
 Distributed under the Boost Software License, Version 1.0. (See accompanying
 file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 =============================================================================*/

// adapter for Python subclass of GlobalMinimizer C++ class

#ifndef INCLUDED_CALLBACK_GLOBAL_MINIMIZATION
#define INCLUDED_CALLBACK_GLOBAL_MINIMIZATION

#include "libgibbs/include/optimizer/utils/simplicial_facet.hpp"
#include "libgibbs/include/optimizer/global_minimization.hpp"
#include <boost/python.hpp>
#include <boost/python/call_method.hpp>

typedef double PyCoordinateType;
typedef std::vector<PyCoordinateType> PyPointType;
typedef Optimizer::details::SimplicialFacet<PyCoordinateType> PyFacetType;
typedef double PyEnergyType;
typedef Optimizer::GlobalMinimizer<PyFacetType,PyCoordinateType,PyEnergyType> PyGlobalMinClass;

// Helper class to dispatch calls to Python subclass of GlobalMinimizer.
// This allows us to subclass GlobalMinimizer from Python, and actually
// affect the behavior inside C++.
struct GlobalMinimizer_callback : public PyGlobalMinClass {
public:
    // Pull protected options into public memory
    using PyGlobalMinClass::initial_subdivisions_per_axis;
    using PyGlobalMinClass::refinement_subdivisions_per_axis;
    using PyGlobalMinClass::critical_edge_length;
    using PyGlobalMinClass::max_search_depth;
    
    GlobalMinimizer_callback(PyObject *p)
    : PyGlobalMinClass(), self(p) {}
    GlobalMinimizer_callback(PyObject *p, PyGlobalMinClass const &x) 
    : PyGlobalMinClass(x), self(p) {}
    
    
    static std::map<std::string,CompositionSet> extract_map_from_dict(
        boost::python::dict const &phase_dict
    )
    {
        using namespace boost::python;
        list keys = phase_dict.keys();
        
        // extract map from Python dict
        std::map<std::string,CompositionSet> phase_list;
        for(int i = 0; i < len(keys); ++i) {
            object curArg = phase_dict[keys[i]];
            if(curArg) {
                phase_list[extract<std::string>(keys[i])] = extract<CompositionSet>(phase_dict[keys[i]]);
            }               
        }
        return phase_list;
    }
    
    std::vector<PyPointType> point_sample(
        CompositionSet const &comp_set,
        sublattice_set const &sublset,
        evalconditions const &conditions
    ) {
        return boost::python::call_method<std::vector<PyPointType>>(self, "point_sample", comp_set, sublset, conditions); 
    }
    std::vector<PyPointType> internal_hull(
        CompositionSet const& comp_set,
        std::vector<PyPointType> const& points,
        std::set<std::size_t> const& dependent_dimensions,
        evalconditions const& conditions
    ) {
        return boost::python::call_method<std::vector<PyPointType>>(self, "internal_hull", comp_set, points, dependent_dimensions, conditions);  
    }
    std::vector<PyFacetType> global_hull(
        std::vector<PyPointType> const& points,
        std::map<std::string,CompositionSet> const& phase_list,
        evalconditions const& conditions
    ) {
        return boost::python::call_method<std::vector<PyFacetType>>(self, "global_hull", points, phase_list, conditions);  
    }
    void run(
        boost::python::dict const &phase_dict,
        sublattice_set const &sublset,
        evalconditions const& conditions
    ) 
    {
        boost::python::call_method<void>(self, "run", phase_dict, sublset, conditions);
    }
    
    static std::vector<PyPointType> default_point_sample(
                                    PyGlobalMinClass &self_, 
                                    CompositionSet const &comp_set,
                                    sublattice_set const &sublset,
                                    evalconditions const &conditions
                                    ) 
    { 
        return self_.PyGlobalMinClass::point_sample(comp_set,sublset,conditions); 
    }
    static void default_run(
        PyGlobalMinClass &self_,
        boost::python::dict const &phase_dict,
        sublattice_set const &sublset,
        evalconditions const& conditions
    ) 
    {
        self_.PyGlobalMinClass::run(extract_map_from_dict(phase_dict), sublset ,conditions); 
    }
    static std::vector<PyPointType> default_internal_hull(
        PyGlobalMinClass &self_, 
        CompositionSet const& cmp,
        std::vector<PyPointType> const& points,
        std::set<std::size_t> const& dependent_dimensions,
        evalconditions const& conditions
    ) 
    { 
        return self_.PyGlobalMinClass::internal_hull(cmp,points,dependent_dimensions,conditions); 
    }
    static std::vector<PyFacetType> default_global_hull(
        PyGlobalMinClass &self_, 
        std::vector<PyPointType> const& points,
        std::map<std::string,CompositionSet> const& phase_list, 
        evalconditions const& conditions
    ) 
    { 
        return self_.PyGlobalMinClass::global_hull(points,phase_list,conditions); 
    }
private:
    PyObject* self;
           
};
#endif