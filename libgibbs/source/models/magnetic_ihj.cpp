/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Model for magnetic ordering according to Inden-Hillert-Jarl (IHJ)
// Reference: Inden, 1976. Hillert and Jarl, 1978.
// Handling for anti-ferromagnetic (AFM) states done according to Hertzman and Sundman, 1982.
// Implementation follows from a review of the IHJ model by W. Xiong et al., 2012.
// Note: This is the IHJ model, not Xiong's model.

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include <string>
#include <sstream>
#include <set>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/tuple/tuple.hpp>

using boost::spirit::utree;
typedef boost::spirit::utree_type utree_type;
using boost::multi_index_container;
using namespace boost::multi_index;

IHJMagneticModel::IHJMagneticModel(
		const std::string &phasename,
		const sublattice_set &subl_set,
		const parameter_set &param_set,
		const double &afm_factor,
		const double &sro_enthalpy_order_fraction
		) : EnergyModel(phasename, subl_set, param_set) {
	model_ast = utree(0);
}
