/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// derivative.cpp -- support for automatic differentiation

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/warning_disable.hpp"
#include "libtdb/include/conditions.hpp"
#include "libtdb/include/exceptions.hpp"
#include "libtdb/include/utils/math_expr.hpp"
#include <boost/spirit/include/support_utree.hpp>

#include <math.h>
#include <string>

boost::spirit::utree const differentiate_utree
(boost::spirit::utree const& ut, evalconditions const& conditions, std::string const& diffvar) {
	typedef boost::spirit::utree utree;
	typedef boost::spirit::utree_type utree_type;
	//std::cout << "deriv_processing " << ut.which() << " tree: " << ut << std::endl;
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
			double res = 0; // storage for the final result
			double lhs = 0; // left-hand side
			double rhs = 0; // right-hand side
			auto lhsiter = ut.end(); // iterator for lhs tree
			auto rhsiter = ut.end(); // iterators for rhs tree
			std::string op;
			//std::cout << "<list>(";
			while (it != end) {
				if ((*it).which() == utree_type::double_type && std::distance(it,end) == 1) {
					// only one element in utree list, and it's a double
					return 0;
				}
				if ((*it).which() == utree_type::string_type) {
					// operator/function
					boost::spirit::utf8_string_range_type rt = (*it).get<boost::spirit::utf8_string_range_type>();
					op = std::string(rt.begin(), rt.end()); // set the symbol
					//std::cout << "OPERATOR: " << op << std::endl;
					if (op == "@") {
						++it;
						double curT = process_utree(*it, conditions).get<double>();
						//std::cout << "curT: " << curT << std::endl;
						++it;
						double lowlimit = process_utree(*it, conditions).get<double>();
						//std::cout << "lowlimit:" << lowlimit << std::endl;
						if (lowlimit == -1) lowlimit = curT; // lowlimit == -1 means no limit
						++it;
						double highlimit = process_utree(*it, conditions).get<double>();

						if (!is_allowed_value(curT)) {
							BOOST_THROW_EXCEPTION(floating_point_error() << str_errinfo("State variable is infinite, subnormal, or not a number"));
						}
						if (!is_allowed_value(lowlimit) || !is_allowed_value(highlimit)) {
							BOOST_THROW_EXCEPTION(floating_point_error() << str_errinfo("State variable limits are infinite, subnormal, or not a number"));
						}
						//std::cout << "highlimit:" << highlimit << std::endl;
						if (highlimit == -1) highlimit = curT+1; // highlimit == -1 means no limit
						if (highlimit <= lowlimit) {
							BOOST_THROW_EXCEPTION(bounds_error() << str_errinfo("Inconsistent bounds on state variable specified. The upper limit <= the lower limit."));
						}
						++it;
						if ((curT >= lowlimit) && (curT < highlimit)) {
							// Range check satisfied
							// Differentiate the tree and return the result
							// Note: by design we only return the first result to satisfy the criterion
							return differentiate_utree(*it, conditions, diffvar).get<double>();
						}
						else {
							// Range check not satisfied
							++it; // Advance to the next token (if any)
							if (it == end) {
								// We are at the end and we failed all range checks
								// The upstream system may decide this is not a problem
								// and use a value of 0, but we want them to have a choice
								BOOST_THROW_EXCEPTION(range_check_error() << str_errinfo("Ranges specified by parameter do not satisfy current system conditions"));
							}
							continue; // Go back to the start of the loop
						}
						//std::cout << "RESULT: " << res << std::endl;
					}
					// This could just be a lone symbol by itself; handle this case
					if ((rt.end() - rt.begin()) == 1) {
						const char* lonesym(rt.begin());
						if (conditions.statevars.find(*lonesym) != conditions.statevars.end()) {
							if (diffvar == std::string(rt.begin(),rt.end())) return 1;
							else return 0;
						}
					}
					++it; // get left-hand side
					// TODO: exception handling
					if (it != end) lhsiter = it;
					++it; // get right-hand side
					if (it != end) rhsiter = it;

					// TODO: fix operators to obey product and chain rules
					// Add derivative operator support to process_utree
					// And many more things before this will actually work...
					if (op == "+") {
						// derivative of sum is sum of derivatives
						if (lhsiter != end) lhs = differentiate_utree(*lhsiter, conditions, diffvar).get<double>();
						if (rhsiter != end) rhs = differentiate_utree(*rhsiter, conditions, diffvar).get<double>();
						res += (lhs + rhs);  // accumulate the result
					}
					else if (op == "-") {
						// derivative of difference is difference of derivatives
						if (lhsiter != end) lhs = differentiate_utree(*lhsiter, conditions, diffvar).get<double>();
						if (ut.size() == 2) lhs = -lhs; // case of negation (unary operator)
						if (rhsiter != end) rhs = differentiate_utree(*rhsiter, conditions, diffvar).get<double>();
						res += (lhs - rhs);
					}
					else if (op == "*") {
						// derivative of product is lhs'rhs + rhs'lhs (product rule)
						double lhs_deriv = differentiate_utree(*lhsiter, conditions, diffvar).get<double>();
						double rhs_deriv = differentiate_utree(*rhsiter, conditions, diffvar).get<double>();
						lhs = process_utree(*lhsiter, conditions).get<double>();
						rhs = process_utree(*rhsiter, conditions).get<double>();
						res += (lhs_deriv * rhs + rhs_deriv * lhs);
					}
					else if (op == "/") {
						// derivative of quotient is (lhs'rhs - rhs'lhs)/(rhs^2) (quotient rule)
						lhs = process_utree(*lhsiter, conditions).get<double>();
						rhs = process_utree(*rhsiter, conditions).get<double>();
						double lhs_deriv = differentiate_utree(*lhsiter, conditions, diffvar).get<double>();
						double rhs_deriv = differentiate_utree(*rhsiter, conditions, diffvar).get<double>();
						if (rhs == 0) {
							BOOST_THROW_EXCEPTION(divide_by_zero_error());
						}
						else {
							res += ((lhs_deriv * rhs - rhs_deriv * lhs) / pow(rhs,2));
						}
					}
					else if (op == "**") {
						// generalized power rule
						// lhs^rhs * (lhs' * (rhs/lhs) + rhs' * ln(lhs))
						lhs = process_utree(*lhsiter, conditions).get<double>();
						rhs = process_utree(*rhsiter, conditions).get<double>();
						double lhs_deriv = differentiate_utree(*lhsiter, conditions, diffvar).get<double>();
						double rhs_deriv = differentiate_utree(*rhsiter, conditions, diffvar).get<double>();
						if (lhs < 0 && (abs(rhs) < 1 && abs(rhs) > 0)) {
							// the result is complex
							// we do not support this (for now)
							BOOST_THROW_EXCEPTION(domain_error() << str_errinfo("Calculated values are not real"));
						}
						res += (pow(lhs, rhs) * (lhs_deriv * (rhs/lhs) + rhs_deriv * log(lhs)));
					}
					else if (op == "ln") {
						lhs = process_utree(*lhsiter, conditions).get<double>();
						if (lhs <= 0) {
							// outside the domain of ln
							BOOST_THROW_EXCEPTION(domain_error() << str_errinfo("Logarithm of nonpositive number is not defined"));
						}
						double lhs_deriv = differentiate_utree(*lhsiter, conditions, diffvar).get<double>();
						res += lhs_deriv / lhs;
					}
					else if (op == "exp") {
						lhs = process_utree(*lhsiter, conditions).get<double>();
						double lhs_deriv = differentiate_utree(*lhsiter, conditions, diffvar).get<double>();
						res += exp(lhs) * lhs_deriv;
					}
					else {
						// a bad symbol made it into our AST
						BOOST_THROW_EXCEPTION(unknown_symbol_error() << str_errinfo("Unknown operator, function or symbol") << specific_errinfo(op));
					}
					//std::cout << "LHS: " << lhs << std::endl;
					//std::cout << "RHS: " << rhs << std::endl;
					lhs = rhs = 0;
					op.erase();
				}
				++it;
			}
			if (!is_allowed_value<double>(res)) {
				BOOST_THROW_EXCEPTION(floating_point_error() << str_errinfo("Calculated value is infinite, subnormal, or not a number"));
			}
			//std::cout << "process_utree returning: " << res << std::endl;
			return utree(res);
			//std::cout << ") ";
			break;
		}
		case utree_type::function_type: {
			break;
		}
		case utree_type::double_type: {
			return 0;
		}
		case utree_type::string_type: {
			boost::spirit::utf8_string_range_type rt = ut.get<boost::spirit::utf8_string_range_type>();
			const char* op(rt.begin());
			if ((rt.end() - rt.begin()) != 1) {
				// throw an exception (bad symbol/state variable)
				BOOST_THROW_EXCEPTION(bad_symbol_error() << str_errinfo("Non-arithmetic (e.g., @) operators or state variables can only be a single character") << specific_errinfo(op));
			}
			if (conditions.statevars.find(*op) != conditions.statevars.end()) {
				if (diffvar == std::string(rt.begin(),rt.end())) return 1;
				else return 0;
			}
			else {
				// throw an exception: undefined state variable
				BOOST_THROW_EXCEPTION(unknown_symbol_error() << str_errinfo("Unknown operator or state variable") << specific_errinfo(op));
			}
			//std::cout << "<operator>:" << op << std::endl;
			break;
		}
	}
	return utree(utree_type::invalid_type);
}
