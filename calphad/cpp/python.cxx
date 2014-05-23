/*=============================================================================
	Copyright (c) 2012-2014 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libtdb/include/database.hpp"
#include "libtdb/include/logging.hpp"
#include "libgibbs/include/conditions.hpp"
#include "libgibbs/include/equilibrium.hpp"
#include <cmath>
#include <boost/python.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;
using namespace Optimizer;

BOOST_PYTHON_MODULE(libcalphadcpp)
{
	init_logging();
	// Wrapper to convert some common std containers
	class_<std::map<std::string,double>>("StdMap")
		.def(map_indexing_suite<std::map<std::string, double> >() )
	;
	class_<std::map<std::string,Optimizer::PhaseStatus>>("PhaseStatusMap")
		.def(map_indexing_suite<std::map<std::string, PhaseStatus> >() )
	;
	// TODO: why do I have charmaps at all? This is a class decl problem
	class_<std::map<char,double>>("StdCharMap")
		.def(map_indexing_suite<std::map<char, double> >() )
	;
	class_<std::vector<std::string>>("StdVector")
		.def(vector_indexing_suite<std::vector<std::string> >() )
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
}
