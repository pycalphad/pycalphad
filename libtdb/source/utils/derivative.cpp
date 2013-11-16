/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// derivative.cpp -- support for a differentiation

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/warning_disable.hpp"
#include "libtdb/include/conditions.hpp"
#include "libtdb/include/exceptions.hpp"
#include "libtdb/include/utils/math_expr.hpp"
#include <boost/spirit/include/support_utree.hpp>
#include <boost/algorithm/string.hpp>

#include <math.h>
#include <string>


// differentiate the utree without variable evaluation
boost::spirit::utree const differentiate_utree(boost::spirit::utree const& ut, std::string const& diffvar) {
	typedef boost::spirit::utree utree;
	typedef boost::spirit::utree_type utree_type;
	switch ( ut.which() ) {
		case utree_type::invalid_type: {
			break;
		}
		case utree_type::nil_type: {
			break;
		}
		case utree_type::list_type: {
			auto it = ut.begin();
			auto end = ut.end();
			auto lhsiter = ut.end(); // iterator for lhs tree
			auto rhsiter = ut.end(); // iterators for rhs tree
			std::string op;
			utree ret_tree, lhs, rhs;

			while (it != end) {
				if ((*it).which() == utree_type::double_type && std::distance(it,end) == 1) {
					// only one element in utree list, and it's a double
					return utree(0);
				}
				if ((*it).which() == utree_type::int_type && std::distance(it,end) == 1) {
					// only one element in utree list, and it's an int
					return utree(0);
				}
				if ((*it).which() == utree_type::string_type) {
					// operator/function
					boost::spirit::utf8_string_range_type rt = (*it).get<boost::spirit::utf8_string_range_type>();
					op = std::string(rt.begin(), rt.end()); // set the symbol
					boost::algorithm::to_upper(op);

					if (op == "@") {
						if (it != end) ++it;
						auto curT = it;
						if (it != end) ++it;
						auto lowlimit = it; // lowlimit
						if (it != end) ++it;
						auto highlimit = it; // highlimit
						if (it != end) ++it;
						utree push_tree = simplify_utree(differentiate_utree(*it, diffvar));
						if (it != end) ++it;
						if (is_zero_tree(push_tree)) {
							// this tree is trivial and this range check operation can be removed
							if (it == end) {
								if (ret_tree.which() == utree_type::invalid_type) {
									// all range checks are trivial; simplify to clean zero
									return utree(0);
								}
								else return ret_tree;
							}
							else {
								// this tree is trivial but we have more range checks to evaluate
								// move on and go back to the top of the loop
								continue;
							}
						}
						else {
							// this is a non-trivial tree and must be returned
							ret_tree.push_back("@");
							ret_tree.push_back(*curT);
							ret_tree.push_back(*lowlimit);
							ret_tree.push_back(*highlimit);
							ret_tree.push_back(push_tree);
							if (it == end) {
								return ret_tree;
							}
							else {
								continue;
							}
						}
					}

					++it; // get left-hand side
					// TODO: exception handling
					if (it != end) lhsiter = it;
					++it; // get right-hand side
					if (it != end) rhsiter = it;

					if (op == "+") {
						// derivative of sum is sum of derivatives
						bool lhszero, rhszero = false;
						if (lhsiter != end) lhs = simplify_utree(differentiate_utree(*lhsiter, diffvar));
						if (rhsiter != end) rhs = simplify_utree(differentiate_utree(*rhsiter, diffvar));
						if (is_zero_tree(lhs)) return simplify_utree(rhs);
						if (is_zero_tree(rhs)) return simplify_utree(lhs);
						ret_tree.push_back("+");
						ret_tree.push_back(lhs);
						ret_tree.push_back(rhs);
						return ret_tree;
					}
					else if (op == "-") {
						// derivative of difference is difference of derivatives
						if (lhsiter != end) lhs = simplify_utree(differentiate_utree(*lhsiter, diffvar));
						if (rhsiter != end) rhs = simplify_utree(differentiate_utree(*rhsiter, diffvar));
						if (is_zero_tree(lhs) && is_zero_tree(rhs)) return utree(0);
						if (ut.size() == 2) {
							if (is_zero_tree(lhs)) return utree(0);
							// case of negation (unary operator)
							ret_tree.push_back("-");
							ret_tree.push_back(lhs);
							return ret_tree;
						}
						if (is_zero_tree(rhs)) return simplify_utree(lhs);
						ret_tree.push_back("-");
						ret_tree.push_back(lhs);
						ret_tree.push_back(rhs);
						return ret_tree;
					}
					else if (op == "*") {
						// derivative of product is lhs'rhs + rhs'lhs (product rule)
						// TODO: optimizations for multiplication by 1 and 0
						utree lhs_deriv = simplify_utree(differentiate_utree(*lhsiter, diffvar));
						utree rhs_deriv = simplify_utree(differentiate_utree(*rhsiter, diffvar));
						utree lhs_prod_tree, rhs_prod_tree;
						lhs_prod_tree.push_back("*");
						lhs_prod_tree.push_back(lhs_deriv);
						lhs_prod_tree.push_back(*rhsiter);
						lhs_prod_tree = simplify_utree(lhs_prod_tree);

						rhs_prod_tree.push_back("*");
						rhs_prod_tree.push_back(rhs_deriv);
						rhs_prod_tree.push_back(*lhsiter);
						rhs_prod_tree = simplify_utree(rhs_prod_tree);

						ret_tree.push_back("+");
						ret_tree.push_back(lhs_prod_tree);
						ret_tree.push_back(rhs_prod_tree);
						return ret_tree;
					}
					else if (op == "/") {
						// derivative of quotient is (lhs'rhs - rhs'lhs)/(rhs^2) (quotient rule)
						// TODO: optimization for identity and 0 operations
						utree lhs_deriv = simplify_utree(differentiate_utree(*lhsiter, diffvar));
						utree rhs_deriv = simplify_utree(differentiate_utree(*rhsiter, diffvar));
						utree lhs_prod_tree, rhs_prod_tree, numerator_tree, power_tree;
						lhs_prod_tree.push_back("*");
						lhs_prod_tree.push_back(lhs_deriv);
						lhs_prod_tree.push_back(*rhsiter);

						lhs_prod_tree = simplify_utree(lhs_prod_tree);

						rhs_prod_tree.push_back("*");
						rhs_prod_tree.push_back(rhs_deriv);
						rhs_prod_tree.push_back(*lhsiter);

						rhs_prod_tree = simplify_utree(rhs_prod_tree);

						numerator_tree.push_back("-");
						numerator_tree.push_back(lhs_prod_tree);
						numerator_tree.push_back(rhs_prod_tree);

						numerator_tree = simplify_utree(numerator_tree);
						// Optimization for zero in numerator
						if (is_zero_tree(numerator_tree)) return numerator_tree;

						power_tree.push_back("**");
						power_tree.push_back(*rhsiter);
						power_tree.push_back(2);

						ret_tree.push_back("/");
						ret_tree.push_back(numerator_tree);
						ret_tree.push_back(power_tree);

						return ret_tree;
					}
					else if (op == "**") {
						if ((*rhsiter).which() == utree_type::int_type || (*rhsiter).which() == utree_type::double_type) {
							// exponent is a constant: power rule
							// power rule + chain rule
							// res += rhs * pow(lhs,rhs-1) * lhs_deriv;
							if (is_zero_tree(*rhsiter)) return utree(0);
							utree lhs_deriv = simplify_utree(differentiate_utree(*lhsiter, diffvar));
							if (rhsiter->get<double>() == 1) return lhs_deriv;
							if (is_zero_tree(lhs_deriv)) return utree(0);
							utree prod_tree, power_tree, exponent_tree;

							exponent_tree.push_back("-");
							exponent_tree.push_back(*rhsiter);
							exponent_tree.push_back(1);

							power_tree.push_back("**");
							power_tree.push_back(*lhsiter);
							power_tree.push_back(exponent_tree);

							prod_tree.push_back("*");
							prod_tree.push_back(*rhsiter);
							prod_tree.push_back(power_tree);

							ret_tree.push_back("*");
							ret_tree.push_back(prod_tree);
							ret_tree.push_back(lhs_deriv);

							return ret_tree;
						}
						else {
							// generalized power rule
							// lhs^rhs * (lhs' * (rhs/lhs) + rhs' * ln(lhs))
							if (is_zero_tree(*lhsiter)) return utree(0);
							utree lhs_deriv = simplify_utree(differentiate_utree(*lhsiter, diffvar));
							utree rhs_deriv = simplify_utree(differentiate_utree(*rhsiter, diffvar));
							utree power_tree, prod_tree1, prod_tree2, div_tree, log_tree, add_tree;


							power_tree.push_back("**");
							power_tree.push_back(*lhsiter);
							power_tree.push_back(*rhsiter);

							div_tree.push_back("/");
							div_tree.push_back(*rhsiter);
							div_tree.push_back(*lhsiter);

							log_tree.push_back("LN");
							log_tree.push_back(*lhsiter);

							prod_tree1.push_back("*");
							prod_tree1.push_back(lhs_deriv);
							prod_tree1.push_back(div_tree);

							prod_tree2.push_back("*");
							prod_tree2.push_back(rhs_deriv);
							prod_tree2.push_back(log_tree);
							if (is_zero_tree(rhs_deriv)) prod_tree2 = utree(0);

							add_tree.push_back("+");
							add_tree.push_back(prod_tree1);
							add_tree.push_back(prod_tree2);
							if (is_zero_tree(lhs_deriv)) add_tree = prod_tree2;

							ret_tree.push_back("*");
							ret_tree.push_back(power_tree);
							ret_tree.push_back(add_tree);

							return ret_tree;
						}
					}
					else if (op == "LN") {
						// res += lhs_deriv / lhs;
						utree lhs_deriv = differentiate_utree(*lhsiter, diffvar);
						if (is_zero_tree(lhs_deriv)) return utree(0);
						ret_tree.push_back("/");
						ret_tree.push_back(lhs_deriv);
						ret_tree.push_back(*lhsiter);

						return ret_tree;
					}
					else if (op == "EXP") {
						// res += exp(lhs) * lhs_deriv;
						if (is_zero_tree(*lhsiter)) return utree(0);
						utree lhs_deriv = differentiate_utree(*lhsiter, diffvar);
						if (is_zero_tree(lhs_deriv)) return utree(0);
						utree exp_tree;
						exp_tree.push_back("EXP");
						exp_tree.push_back(*lhsiter);

						ret_tree.push_back("*");
						ret_tree.push_back(exp_tree);
						ret_tree.push_back(lhs_deriv);

						return ret_tree;
					}

					// not an operator, must be a symbol
					if (op == diffvar) return utree(1);
					else return utree(0);
				}
				++it;
			}
			break;
		}
		case utree_type::function_type: {
			break;
		}
		case utree_type::double_type: {
			return utree(0);
		}
		case utree_type::int_type: {
			return utree(0);
		}
		case utree_type::string_type: {
			boost::spirit::utf8_string_range_type rt = ut.get<boost::spirit::utf8_string_range_type>();
			std::string varname(rt.begin(),rt.end());

			if (diffvar == varname) return utree(1);
			else return utree(0);
		}
	}
	BOOST_THROW_EXCEPTION(unknown_symbol_error() << str_errinfo("Unable to differentiate abstract syntax tree") << ast_errinfo(ut));
	return utree();
}
