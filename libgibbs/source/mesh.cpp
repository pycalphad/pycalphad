/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/mesh.hpp"
#include "libtdb/include/exceptions.hpp"


Mesh::Mesh(const evalconditions &conds) : startpoint(conds) { }; // init Mesh with starting point
MeshAxis::MeshAxis() { }
MeshAxis::MeshAxis(const double &argmin, const double &argmax, const double &subint, const MeshAxisType &type) :
		min(argmin), max(argmax), subinterval(subint), axistype(type) {}

void Mesh::SetMeshAxis(const std::string &var, const double &min, const double &max, const double &subinterval, const MeshAxisType &axtype) {
	if (min >= max) {
		BOOST_THROW_EXCEPTION(range_check_error() << str_errinfo("Mesh point spacing must be positive"));
	}
	// TODO: check if distance between max and min is less than subinterval; TRICK: dependent on MeshAxisType
	// TODO: syntax checking for the condition variable input
	axes[var] = MeshAxis(min, max, subinterval, axtype);
}

void Mesh::SetMeshAxis(const std::string &var, const double &min, const double &max, const double &subinterval) {
	SetMeshAxis(var, min, max, subinterval, MeshAxisType::LINEAR); // assume linear spacing
}

void Mesh::SetMeshAxis(const std::string &var, const double &min, const double &max) {
	double default_density = 20; // TODO: make default linear density an option
	double interval = (max-min)/default_density;
	if (interval <= 0) {
		BOOST_THROW_EXCEPTION(range_check_error() << str_errinfo("Mesh point spacing must be positive"));
	}
	SetMeshAxis(var, min, max, interval, MeshAxisType::LINEAR);
}
