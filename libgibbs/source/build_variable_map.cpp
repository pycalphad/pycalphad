/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

	Based on example code from Boost MultiIndex.
	Copyright (c) 2003-2008 Joaquin M Lopez Munoz.
	See http://www.boost.org/libs/multi_index for library home page.

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// Helper functions for AST-based models

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/models.hpp"
#include "libgibbs/include/optimizer/opt_Gibbs.hpp"
#include "libtdb/include/logging.hpp"
#include <string>
#include <map>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>

using boost::spirit::utree;
typedef boost::spirit::utree_type utree_type;
using boost::multi_index_container;
using namespace boost::multi_index;

sublattice_set build_variable_map(
		const Phase_Collection::const_iterator p_begin,
		const Phase_Collection::const_iterator p_end,
		const evalconditions &conditions,
		std::map<std::string, int> &indices
		) {
	logger opt_log(journal::keywords::channel = "optimizer");
	sublattice_set ret_set;

	int indexcount = 0; // counter for variable indices (for optimizer)

	for (auto i = conditions.elements.cbegin(); i != conditions.elements.cend(); ++i) {
		BOOST_LOG_SEV(opt_log, debug) << "conditions element: " << (*i);
	}

	// All phases
	for (auto i = p_begin; i != p_end; ++i) {
		auto const cond_find = conditions.phases.find(i->first);
		if (cond_find->second != PhaseStatus::ENTERED) continue;
		auto subl_start = i->second.get_sublattice_iterator();
		auto subl_end = i->second.get_sublattice_iterator_end();
		if (subl_start == subl_end) BOOST_LOG_SEV(opt_log, critical) << "No sublattices found!";
		std::string phasename = i->first;

		indices[phasename + "_FRAC"] = indexcount; // save index of phase fraction
		// insert fake record for the phase fraction variable at -1 sublattice index

		ret_set.insert(sublattice_entry(-1, indexcount++, 0, phasename, ""));
		BOOST_LOG_SEV(opt_log, debug) << "building phase " << phasename;
		// All sublattices
		for (auto j = subl_start; j != subl_end;++j) {
			// All species
			BOOST_LOG_SEV(opt_log, debug) << "looping sublattice " << std::distance(subl_start,j);
			for (auto k = (*j).get_species_iterator(); k != (*j).get_species_iterator_end();++k) {
				BOOST_LOG_SEV(opt_log, debug) << "checking species " << (*k);
				// Check if this species in this sublattice is on our list of elements to investigate
				if (std::find(conditions.elements.cbegin(),conditions.elements.cend(),*k) != conditions.elements.cend()) {
					BOOST_LOG_SEV(opt_log, debug) << "matched species " << (*k);
					int sublindex = std::distance(subl_start,j);
					double sitecount = (*j).stoi_coef;
					std::string spec = (*k);
					std::stringstream varname;
					varname << phasename << "_" << sublindex << "_" << spec; // build variable name
					indices[varname.str()] = indexcount; // save index of variable
					ret_set.insert(sublattice_entry(sublindex, indexcount++, sitecount, phasename, spec));
					BOOST_LOG_SEV(opt_log, debug) << "inserted sublattice_entry for " << varname.str();
				}
			}
		}
	}
	return ret_set;
}
