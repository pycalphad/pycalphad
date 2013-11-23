/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition for CompositionSet class

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/compositionset.hpp"

CompositionSet::CompositionSet(const Phase &phaseobj, const parameter_set &pset, const sublattice_set &sublset) {
	cset_name = phaseobj.name();

	// Now initialize the appropriate models
	models["PURE_ENERGY"] = std::unique_ptr<EnergyModel>(new PureCompoundEnergyModel(phaseobj.name(), sublset, pset));
	models["IDEAL_MIX"] = std::unique_ptr<EnergyModel>(new IdealMixingModel(phaseobj.name(), sublset));
	models["REDLICH_KISTER"] = std::unique_ptr<EnergyModel>(new RedlichKisterExcessEnergyModel(phaseobj.name(), sublset, pset));
	models["IHJ_MAGNETIC"] = std::unique_ptr<EnergyModel>(new IHJMagneticModel(phaseobj.name(), sublset, pset,
					phaseobj.magnetic_afm_factor, phaseobj.magnetic_sro_enthalpy_order_fraction));
}
