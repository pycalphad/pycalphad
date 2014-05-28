/*=============================================================================
	Copyright (c) 2012-2014 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libtdb/include/structure.hpp"
#include "libtdb/include/parameter.hpp"
#include "libtdb/include/database.hpp"
#include "libtdb/include/logging.hpp"
#include "libgibbs/include/conditions.hpp"
#include "libgibbs/include/compositionset.hpp"
#include "libgibbs/include/equilibrium.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/utils/build_variable_map.hpp"
#include "python_ext/callback_global_minimization.hpp"
#include <cmath>
#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;
using namespace Optimizer;

BOOST_PYTHON_MODULE(libcalphadcpp)
{
    typedef PyGlobalMinClass::HullMapType::HullEntryType PyConvexHullEntry;
    typedef PyGlobalMinClass::HullMapType::HullEntryContainerType PyConvexHullEntries;
    init_logging();
    // Wrapper to convert some common std containers
    class_<std::map<std::string,double>>("StdMap")
            .def(map_indexing_suite<std::map<std::string, double> >() )
    ;
    class_<std::map<std::string,int>>("IndexStdMap")
    .def(map_indexing_suite<std::map<std::string, int> >() )
    ;
    class_<boost::bimap<std::string,int>>("IndexBiMap")
    ;
    class_<std::map<std::string,Optimizer::PhaseStatus>>("PhaseStatusMap")
            .def(map_indexing_suite<std::map<std::string, PhaseStatus> >() )
    ;
    class_<std::map<std::string,::Phase>>("PhaseMap")
    .def(map_indexing_suite<std::map<std::string, ::Phase> >() )
    ;
    class_<std::map<std::string,CompositionSet>>("CompositionSetMap", no_init)
    .def(map_indexing_suite<std::map<std::string, CompositionSet>>() )
    ;
    // TODO: why do I have charmaps at all? This is a class decl problem
    class_<std::map<char,double>>("StdCharMap")
            .def(map_indexing_suite<std::map<char, double> >() )
    ;
    class_<std::vector<std::string>>("StdVector")
            .def(vector_indexing_suite<std::vector<std::string> >() )
    ;
    class_<std::vector<std::size_t>>("SizeVector")
    .def(vector_indexing_suite<std::vector<std::size_t> >() )
    ;
    enum_<PhaseStatus>("PhaseStatus")
            .value("ENTERED", PhaseStatus::ENTERED)
            .value("DORMANT", PhaseStatus::DORMANT)
            .value("FIXED", PhaseStatus::FIXED)
            .value("SUSPENDED",  PhaseStatus::SUSPENDED)
    ;

    class_<evalconditions>("evalconditions")
    	.def_readwrite("statevars", &evalconditions::statevars)
    	.def_readwrite("elements", &evalconditions::elements)
    	.def_readwrite("phases", &evalconditions::phases)
    	.def_readwrite("xfrac", &evalconditions::xfrac)
    ;
    
    class_<Database>("Database")
    .def(init<std::string>()) // alternative constructor
    .def("get_info", &Database::get_info)
    .def("get_phases", &Database::get_phases)
    .def("get_parameter_set", &Database::get_parameter_set)
    .def("process_command", &Database::proc_command)
    ;

    class_<Equilibrium, boost::shared_ptr<Equilibrium>, boost::noncopyable>("Equilibrium", no_init)
    	.def("__repr__", &Equilibrium::print)
    	.def("__str__", &Equilibrium::print)
    	.def("GibbsEnergy", &Equilibrium::GibbsEnergy)
    ;
    class_<EquilibriumFactory, boost::noncopyable>("EquilibriumFactory")
        .def("create", &EquilibriumFactory::create)
    ; 
    class_<::Phase>("Phase")
    // missing implementation .def_readwrite("subls", &::Phase::subls)
    .def_readwrite("init_cmds", &::Phase::init_cmds)
    .def_readwrite("magnetic_afm_factor", &::Phase::magnetic_afm_factor)
    .def_readwrite("magnetic_sro_enthalpy_order_fraction", &::Phase::magnetic_sro_enthalpy_order_fraction)
    .def_readwrite("name", &::Phase::phase_name)
    ;
    class_<sublattice_entry>("sublattice_entry", init<int,int,double,std::string,std::string>() )
    .def_readwrite("index", &sublattice_entry::index)
    .def_readwrite("opt_index", &sublattice_entry::opt_index)
    .def_readwrite("phase", &sublattice_entry::phase)
    .def_readwrite("species", &sublattice_entry::species)
    ;
    class_<sublattice_set>("sublattice_set")
    ;
    class_<parameter_set>("parameter_set")
    ;
    // function pointer for overloaded build_variable_map()
    sublattice_set (*bvm1)( 
         const Phase_Collection&, 
         const evalconditions&,
         boost::bimap<std::string,int>&
                          ) = &build_variable_map;
    def("build_variable_map", bvm1)
    ;

    class_<CompositionSet>("CompositionSet", 
                           init<const ::Phase&,
                                const parameter_set&, 
                                const sublattice_set&, 
                                const boost::bimap<std::string,int>&
                                >()
                          )
    .def(init<const CompositionSet&,const std::map<std::string, double>&,const std::string&> () )
    .def("name", &CompositionSet::name)
    ;
    class_<PyConvexHullEntry>("ConvexHullEntry")
    .def_readwrite("phase_name", &PyConvexHullEntry::phase_name)
    .def_readwrite("energy", &PyConvexHullEntry::energy)
    .def_readwrite("internal_coordinates", &PyConvexHullEntry::internal_coordinates)
    .def_readwrite("global_coordinates", &PyConvexHullEntry::global_coordinates)
    ;
    class_<PyConvexHullEntries>("ConvexHullEntries")
    .def(vector_indexing_suite<PyConvexHullEntries>() )
    ;
    class_<PyFacetType>("Facet")
    .def_readwrite("area", &PyFacetType::area)
    .def_readwrite("normal", &PyFacetType::normal)
    .def_readwrite("vertices", &PyFacetType::vertices)
    //TODO .def_readwrite("basis_matrix", &PyFacetType::basis_matrix)
    ;
    class_<std::vector<PyFacetType>>("Facets")
    .def(vector_indexing_suite<std::vector<PyFacetType>>() )
    ;
    
    class_<PyGlobalMinClass,GlobalMinimizer_callback>("GlobalMinimizer",
                                                    init<const boost::python::dict&,
                                                        const sublattice_set&,
                                                        const evalconditions&
                                                        >()
    )
    .def("get_hull_entries", &GlobalMinimizer_callback::get_hull_entries)
    .def("get_facets", &GlobalMinimizer_callback::get_facets)
    ;
}
