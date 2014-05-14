/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

#ifndef MESH_INCLUDED
#define MESH_INCLUDED

// declaration for Mesh object

#include <unordered_map>
#include <string>
#include "libgibbs/include/conditions.hpp"


enum class MeshAxisType : unsigned int {
	LINEAR = 1, LOGARITHMIC = 2, INVERSE = 3, CUSTOM = 4 // TODO: CUSTOM will rely on a specified function
};

struct MeshAxis {
	double min; // minimum value of the partition
	double max; // maximum value of the partition
	double subinterval; // MeshAxisType distance between points
	MeshAxisType axistype; // transform the variable to this form before partitioning
	MeshAxis();
	MeshAxis(const double &argmin, const double &argmax, const double &subint, const MeshAxisType &type);
};

class Mesh {
private:
	// collection of evalconditions objects
	const evalconditions startpoint;
	std::unordered_map<std::string,MeshAxis> axes;
public:
	Mesh(const evalconditions &);
	void SetMeshAxis(const std::string &var, const double &min, const double &max, const double &subinterval, const MeshAxisType &);
	void SetMeshAxis(const std::string &var, const double &min, const double &max, const double &subinterval);
	void SetMeshAxis(const std::string &var, const double &min, const double &max);
};
#endif
