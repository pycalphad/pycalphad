/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

	Based on example code from Boost MultiIndex.
	Copyright (c) 2003-2008 Joaquin M Lopez Munoz.
	See http://www.boost.org/libs/multi_index for library home page.

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// header for Gibbs energy model structures
#ifndef INCLUDED_MODELS_HPP
#define INCLUDED_MODELS_HPP

#include "libtdb/include/structure.hpp"
#include "libgibbs/include/utils/ast_caching.hpp"
#include <boost/spirit/include/support_utree.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/range/irange.hpp>
#include <string>
#include <sstream>


struct sublattice_entry {
	int index; // sublattice index
	int opt_index; // variable index (for optimizer)
	double num_sites; // number of sites
	std::string phase;
	std::string species; // species name
	sublattice_entry (
			int index_, int opt_index_, double num_sites_, std::string phase_, std::string species_) :
				index(index_),
				opt_index(opt_index_),
				num_sites(num_sites_),
				phase(phase_),
				species(species_) {}
	const std::string name() const {
		// std::to_string exists in C++11 but some compilers are buggy
		std::stringstream ss;
		ss << phase << "_" << index << "_" << species;
		return (const std::string)ss.str();
	}
};

/* Tags for multi-indexing */
struct myindex{};
struct optimizer_index{};
struct phases{};
struct phase_subl{};

/* Sublattice entries are sorted first by phase name, then index, then by species.
 * NB: The use of derivation here instead of simple typedef is explained in
 * Compiler specifics: type hiding.
 */

struct subl_sort_key : boost::multi_index::composite_key<
sublattice_entry,
BOOST_MULTI_INDEX_MEMBER(sublattice_entry, std::string,phase),
BOOST_MULTI_INDEX_MEMBER(sublattice_entry,int,index),
BOOST_MULTI_INDEX_MEMBER(sublattice_entry,std::string,species)
>{};


/* see Compiler specifics: composite_key in compilers without partial
 * template specialization, for info on composite_key_result_less
 */

typedef boost::multi_index::multi_index_container<
		sublattice_entry,
		boost::multi_index::indexed_by<
		boost::multi_index::ordered_non_unique<
		boost::multi_index::tag<myindex>,
		BOOST_MULTI_INDEX_MEMBER(sublattice_entry,int,index)
		>,
		boost::multi_index::ordered_non_unique<
		boost::multi_index::tag<optimizer_index>,
		BOOST_MULTI_INDEX_MEMBER(sublattice_entry,int,opt_index)
		>,
		boost::multi_index::ordered_non_unique<
		boost::multi_index::tag<phases>,
		BOOST_MULTI_INDEX_MEMBER(sublattice_entry,std::string,phase)
		>,
		boost::multi_index::ordered_non_unique<
		boost::multi_index::tag<phase_subl>,
		boost::multi_index::composite_key<
			sublattice_entry,
			BOOST_MULTI_INDEX_MEMBER(sublattice_entry,std::string,phase),
			BOOST_MULTI_INDEX_MEMBER(sublattice_entry,int,index)
			>
>
>
> sublattice_set;


typedef boost::multi_index::multi_index_container<
		const sublattice_entry*,
		boost::multi_index::indexed_by<
		boost::multi_index::ordered_non_unique<
		boost::multi_index::tag<myindex>,
		BOOST_MULTI_INDEX_MEMBER(sublattice_entry,const int,index)
		>,
		boost::multi_index::ordered_non_unique<
		boost::multi_index::tag<optimizer_index>,
		BOOST_MULTI_INDEX_MEMBER(sublattice_entry,const int,opt_index)
		>,
		boost::multi_index::ordered_non_unique<
		boost::multi_index::tag<phases>,
		BOOST_MULTI_INDEX_MEMBER(sublattice_entry,const std::string,phase)
		>,
		boost::multi_index::ordered_non_unique<
		boost::multi_index::tag<phase_subl>,
		boost::multi_index::composite_key<
			sublattice_entry,
			BOOST_MULTI_INDEX_MEMBER(sublattice_entry,std::string,phase),
			BOOST_MULTI_INDEX_MEMBER(sublattice_entry,int,index)
			>
>
>
> sublattice_set_view;


// prototype model
class EnergyModel {
public:
	typedef std::map<std::string, CachedAbstractSyntaxTree> ASTSymbolMap;
	EnergyModel(const std::string &phasename, const sublattice_set &subl_set, const parameter_set &param_set) {
		// implementation
	}
	EnergyModel(const std::string &phasename, const sublattice_set &subl_set) {
		// implementation
	};
	const boost::spirit::utree& get_ast() const { return model_ast; }
	const boost::iterator_range<ASTSymbolMap::const_iterator> get_symbol_table() const {
		return boost::make_iterator_range(ast_symbol_table.begin(), ast_symbol_table.end());
	}
protected:
	boost::spirit::utree model_ast;
	ASTSymbolMap ast_symbol_table; // storage for expensive, repeating ASTs behind a symbol
	double count_mixing_sites(const sublattice_set_view &ssv);
	boost::spirit::utree add_interaction_factor (
			const std::string &lhs_varname,
			const std::string &rhs_varname,
			const double &degree,
			const boost::spirit::utree &input_tree
			);
	void normalize_utree(boost::spirit::utree &input_tree, const sublattice_set_view &ssv);
	boost::spirit::utree find_parameter_ast(const sublattice_set_view &subl_view, const parameter_set_view &param_view);
	boost::spirit::utree permute_site_fractions (
			const sublattice_set_view &total_view, // all sublattices
			const sublattice_set_view &subl_view, // the active sublattice permutation
			const parameter_set_view &param_view,
			const int &sublindex,
			const double &param_division_factor = 1
			);
	boost::spirit::utree permute_site_fractions_with_interactions (
			const sublattice_set_view &total_view, // all sublattices
			const sublattice_set_view &subl_view, // the active sublattice permutation
			const parameter_set_view &param_view,
			const int &sublindex,
			const double &param_division_factor = 1
			);
private:
	EnergyModel(const EnergyModel&);
	EnergyModel& operator=(const EnergyModel&);
};


// default models
class PureCompoundEnergyModel : public EnergyModel {
public:
	PureCompoundEnergyModel(
			const std::string &phasename,
			const sublattice_set &subl_set,
			const parameter_set &param_set
			);
};

class IdealMixingModel : public EnergyModel {
public:
	IdealMixingModel(
			const std::string &phasename,
			const sublattice_set &subl_set
			);
protected:
	void protect_domain(const std::string &varname, boost::spirit::utree &input_tree);
	boost::spirit::utree make_xlnx(const std::string &varname);
};

class RedlichKisterExcessEnergyModel : public EnergyModel {
public:
	RedlichKisterExcessEnergyModel(
			const std::string &phasename,
			const sublattice_set &subl_set,
			const parameter_set &param_set
			);
};

class IHJMagneticModel : public EnergyModel {
public:
	IHJMagneticModel(
			const std::string &phasename,
			const sublattice_set &subl_set,
			const parameter_set &param_set,
			const double &afm_factor,
			const double &sro_enthalpy_order_fraction
			);
};

#endif
