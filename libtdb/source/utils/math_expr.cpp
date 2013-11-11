/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// math_expr.cpp -- evaluator for Thermo-Calc-like mathematical expressions

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/warning_disable.hpp"
#include "libtdb/include/conditions.hpp"
#include "libtdb/include/exceptions.hpp"
#include "libtdb/include/utils/math_expr.hpp"
#include <boost/spirit/include/support_utree.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/algorithm/string.hpp>

#include <math.h>
#include <string>

using boost::spirit::utree;
using boost::spirit::utree_type;

// Determine if the given floating-point value is allowed in our calculation
template <typename T>
bool is_allowed_value(T &val) {
	int fpclass = boost::math::fpclassify(val);
	// FP_NORMAL means we have a non-zero, finite, non-subnormal number
	// FP_ZERO means we have zero
	if (fpclass == (int)FP_NORMAL || fpclass == (int)FP_ZERO) {
		return true;
	}
	else {
		return false;
	}
}
template bool is_allowed_value(double&); // explicit instantiation

bool is_zero_tree(const utree &ut) {
	bool condition = ((ut.which() == utree_type::double_type || ut.which() == utree_type::int_type) && ut.get<double>() == 0.0);
	return condition;
}

boost::spirit::utree const process_utree(
		boost::spirit::utree const& ut,
		evalconditions const& conditions,
		std::map<std::string, int> const &modelvar_indices,
		double* const modelvars) {
	typedef boost::spirit::utree utree;
	typedef boost::spirit::utree_type utree_type;
	//std::cout << "processing " << ut.which() << " tree: " << ut << std::endl;
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
			std::string op;
			//std::cout << "<list>(";
			while (it != end) {
				if ((*it).which() == utree_type::double_type && std::distance(it,end) == 1) {
					// only one element in utree list, and it's a double
					// return its value
					double retval = (*it).get<double>();
					if (!is_allowed_value<double>(retval)) {
						BOOST_THROW_EXCEPTION(floating_point_error() << str_errinfo("Calculated value is infinite, subnormal, or not a number") << ast_errinfo(ut));
					}
					return retval;
				}
				if ((*it).which() == utree_type::int_type && std::distance(it,end) == 1) {
					// only one element in utree list, and it's an int
					// return its value
					double retval = (*it).get<double>();
					if (!is_allowed_value<double>(retval)) {
						BOOST_THROW_EXCEPTION(floating_point_error() << str_errinfo("Calculated value is infinite, subnormal, or not a number") << ast_errinfo(ut));
					}
					return retval;
				}
				if ((*it).which() == utree_type::string_type) {
					// operator/function
					boost::spirit::utf8_string_range_type rt = (*it).get<boost::spirit::utf8_string_range_type>();
					op = std::string(rt.begin(), rt.end()); // set the symbol
					boost::algorithm::to_upper(op);
					const auto varindex = modelvar_indices.find(op); // attempt to find this symbol as a model variable
					if (varindex != modelvar_indices.end()) {
						// we found the variable
						// use the index to return the current value
						//std::cout << "variable " << op << " found in list string_type" << std::endl;
						//std::cout << "process_utree returning: " << modelvars[varindex->second] << std::endl;
						return modelvars[varindex->second];
					}
					//std::cout << "OPERATOR: " << op << std::endl;
					if (op == "@") {
						++it;
						double curT, lowlimit, highlimit;
						try {
							curT = process_utree(*it, conditions, modelvar_indices, modelvars).get<double>();
							//std::cout << "curT: " << curT << std::endl;
							++it;
							lowlimit = process_utree(*it, conditions, modelvar_indices, modelvars).get<double>();
							//std::cout << "lowlimit:" << lowlimit << std::endl;
							//if (lowlimit == -1) lowlimit = curT; // lowlimit == -1 means no limit
							++it;
							highlimit = process_utree(*it, conditions, modelvar_indices, modelvars).get<double>();
						}
						catch (boost::exception &e) {
							e << ast_errinfo(*it);
							throw;
						}

						if (!is_allowed_value<double>(curT)) {
							BOOST_THROW_EXCEPTION(floating_point_error() << str_errinfo("Variable is infinite, subnormal, or not a number") << ast_errinfo(ut));
						}
						if (!is_allowed_value<double>(lowlimit) || !is_allowed_value<double>(highlimit)) {
							BOOST_THROW_EXCEPTION(floating_point_error() << str_errinfo("Variable limits are infinite, subnormal, or not a number") << ast_errinfo(ut));
						}
						//std::cout << "highlimit:" << highlimit << std::endl;
						//if (highlimit == -1) highlimit = curT+1; // highlimit == -1 means no limit
						if (highlimit <= lowlimit) {
							BOOST_THROW_EXCEPTION(bounds_error() << str_errinfo("Inconsistent bounds on variable specified. The upper limit <= the lower limit.") << ast_errinfo(ut));
						}
						++it;
						if ((curT >= lowlimit) && (curT < highlimit)) {
							// Range check satisfied
							// Process the tree and return the result
							// Note: by design we only return the first result to satisfy the criterion
							utree ret_tree;
							try {
								ret_tree = process_utree(*it, conditions, modelvar_indices, modelvars).get<double>();
							}
							catch (boost::exception &e) {
								e << ast_errinfo(*it);
								throw;
							}
							return ret_tree;
						}
						else {
							// Range check not satisfied
							++it; // Advance to the next token (if any)
							if (it == end) {
								// Failed all range checks
								return utree(0);
							}
							continue; // Go back to the start of the loop
						}
						//std::cout << "RESULT: " << res << std::endl;
					}
					// This could just be a lone symbol by itself; handle this case
					if ((rt.end() - rt.begin()) == 1) {
						const char* lonesym(rt.begin());
						if (conditions.statevars.find(*lonesym) != conditions.statevars.end()) {
							return conditions.statevars.find(*lonesym)->second; // return current value of state variable (T, P, etc)
						}
					}
					++it; // get left-hand side
					// TODO: exception handling
					auto lhsiter = it;
					if (it != end) {
						try {
							lhs = process_utree(*it, conditions, modelvar_indices, modelvars).get<double>();
						}
						catch (boost::exception &e) {
							//e << ast_errinfo(*it);
							throw;
						}
					}
					++it; // get right-hand side
					if (it != end) {
						try {
							rhs = process_utree(*it, conditions, modelvar_indices, modelvars).get<double>();
						}
						catch (boost::exception &e) {
							//e << ast_errinfo(*it);
							throw;
						}
					}


					if (op == "+") res += (lhs + rhs);  // accumulate the result
					else if (op == "-") {
						if (ut.size() == 2) lhs = -lhs; // case of negation (unary operator)
						res += (lhs - rhs);
					}
					else if (op == "*") res += (lhs * rhs); 
					else if (op == "/") { 
						if (rhs == 0) {
							return utree();
							BOOST_THROW_EXCEPTION(divide_by_zero_error() << ast_errinfo(ut));
						}
						else res += (lhs / rhs);
					}
					else if (op == "**") {
						if (lhs < 0 && (fabs(rhs) < 1 && fabs(rhs) > 0)) {
							// the result is complex
							// we do not support this (for now)
							BOOST_THROW_EXCEPTION(domain_error() << str_errinfo("Calculated values are not real") << ast_errinfo(ut));
						}
						res += pow(lhs, rhs);
					}
					else if (op == "LN") {
						if (lhs > 0) {
							res += log(lhs);
						}
						else {
							// outside the domain of ln
							return utree();
							BOOST_THROW_EXCEPTION(domain_error() << str_errinfo("Logarithm of nonpositive number is not defined") << ast_errinfo(ut));
						}
					}
					else if (op == "EXP") res += exp(lhs);
					else {
						// a bad symbol made it into our AST
						BOOST_THROW_EXCEPTION(unknown_symbol_error() << str_errinfo("Unknown operator, function or symbol") << specific_errinfo(op) << ast_errinfo(ut));
					}
					//std::cout << "LHS: " << lhs << std::endl;
					//std::cout << "RHS: " << rhs << std::endl;
					lhs = rhs = 0;
					op.erase();
				}
				++it;
			}
			if (!is_allowed_value<double>(res)) {
				BOOST_THROW_EXCEPTION(floating_point_error() << str_errinfo("Calculated value is infinite, subnormal, or not a number") << ast_errinfo(ut));
			}
			//std::cout << "process_utree " << ut << " = " << res << std::endl;
			return utree(res);
			//std::cout << ") ";
			break;
		}
		case utree_type::function_type: {
			break;
		}
		case utree_type::double_type: {
			double retval = ut.get<double>();
			if (!is_allowed_value<double>(retval)) {
				BOOST_THROW_EXCEPTION(floating_point_error() << str_errinfo("Calculated value is infinite, subnormal, or not a number") << ast_errinfo(ut));
			}
			return retval;
		}
		case utree_type::int_type: {
			double retval = ut.get<double>();
			if (!is_allowed_value<double>(retval)) {
				BOOST_THROW_EXCEPTION(floating_point_error() << str_errinfo("Calculated value is infinite, subnormal, or not a number") << ast_errinfo(ut));
			}
			return retval;
		}
		case utree_type::string_type: {
			boost::spirit::utf8_string_range_type rt = ut.get<boost::spirit::utf8_string_range_type>();
			std::string varname(rt.begin(),rt.end());

			const auto varindex = modelvar_indices.find(varname); // attempt to find this variable
			if (varindex != modelvar_indices.end()) {
				// we found the variable
				// use the index to return the current value
				//std::cout << "var found in string_type = " << modelvars[varindex->second] << std::endl;
				//std::cout << "process_utree returning: " << modelvars[varindex->second] << std::endl;
				return modelvars[varindex->second];
			}
			const char* op(rt.begin());
			if ((rt.end() - rt.begin()) != 1) {
				// throw an exception (bad symbol/state variable)
				BOOST_THROW_EXCEPTION(bad_symbol_error() << str_errinfo("Non-arithmetic (e.g., @) operators or state variables can only be a single character") << specific_errinfo(op) << ast_errinfo(ut));
			}
			if (conditions.statevars.find(*op) != conditions.statevars.end()) {
				//std::cout << "T executed" << std::endl;
				//std::cout << "process_utree returning: " << conditions.statevars.find(*op)->second << std::endl;
				return conditions.statevars.find(*op)->second; // return current value of state variable (T, P, etc)
			}
			else {
				// throw an exception: undefined state variable
				BOOST_THROW_EXCEPTION(unknown_symbol_error() << str_errinfo("Unknown operator or state variable") << specific_errinfo(op) << ast_errinfo(ut));
			}
			//std::cout << "<operator>:" << op << std::endl;
			break;
		}
	}
	return utree();
}

boost::spirit::utree const simplify_utree(boost::spirit::utree const& ut) {
	typedef boost::spirit::utree utree;
	typedef boost::spirit::utree_type utree_type;
	//std::cout << "processing " << ut.which() << " tree: " << ut << std::endl;
	switch ( ut.which() ) {
		case utree_type::invalid_type: {
			return ut;
			break;
		}
		case utree_type::nil_type: {
			return ut;
			break;
		}
		case utree_type::list_type: {
			auto it = ut.begin();
			auto end = ut.end();
			double res = 0; // storage for the final result
			std::string op;
			//std::cout << "<list>(";
			while (it != end) {
				if ((*it).which() == utree_type::double_type && std::distance(it,end) == 1) {
					// only one element in utree list, and it's a double
					// return its value
					double retval = (*it).get<double>();
					if (!is_allowed_value<double>(retval)) {
						return ut;
					}
					return retval;
				}
				if ((*it).which() == utree_type::int_type && std::distance(it,end) == 1) {
					// only one element in utree list, and it's an int
					// return its value
					double retval = (*it).get<double>();
					if (!is_allowed_value<double>(retval)) {
						return ut;
					}
					return retval;
				}
				if ((*it).which() == utree_type::string_type) {
					// operator/function
					boost::spirit::utf8_string_range_type rt = (*it).get<boost::spirit::utf8_string_range_type>();
					utree lhs, rhs;
					op = std::string(rt.begin(), rt.end()); // set the symbol
					boost::algorithm::to_upper(op);
					//std::cout << "OPERATOR: " << op << std::endl;
					if (op == "@") {
						if (it != end) ++it; // curT
						if (it != end) ++it; // lowlimit
						if (it != end) ++it; // highlimit
						if (it != end) ++it; // tree
						if (it != end) {
							// this should be the range-specific tree
							// if it's a trivial zero, we should discard this operation
							//bool checkzero = is_zero_tree(simplify_utree(*it));
							//if (checkzero && it == end) return utree(0);
							//if (checkzero && it != end) continue;
						}
						return ut;
					}
					++it; // get left-hand side
					// TODO: exception handling
					if (it != end) lhs = simplify_utree(*it);
					++it; // get right-hand side
					if (it != end) rhs = simplify_utree(*it);

					if (op == "+") {
						if ((lhs.which() == utree_type::double_type || lhs.which() == utree_type::int_type)
								&&
								(rhs.which() == utree_type::double_type || rhs.which() == utree_type::int_type)
						) {
							res += (lhs.get<double>() + rhs.get<double>());  // accumulate the result
						}
						else return ut;
					}
					else if (op == "-") {
						if (ut.size() == 2) {
							if (lhs.which() == utree_type::double_type || lhs.which() == utree_type::int_type) {
								res += -lhs.get<double>(); // case of negation (unary operator)
							}
							else return ut;
						}
						else {
							if ((lhs.which() == utree_type::double_type || lhs.which() == utree_type::int_type)
									&&
									(rhs.which() == utree_type::double_type || rhs.which() == utree_type::int_type)
							) {
								res += (lhs.get<double>() - rhs.get<double>());
							}
							else return ut;
						}
					}
					else if (op == "*") {
						if (is_zero_tree(lhs))
							res += 0;
						else if (is_zero_tree(rhs))
							res += 0;
						else if ((lhs.which() == utree_type::double_type || lhs.which() == utree_type::int_type)
								&&
								(rhs.which() == utree_type::double_type || rhs.which() == utree_type::int_type)
						) {
							res += (lhs.get<double>() * rhs.get<double>());
						}
						else return ut;
					}
					else if (op == "/") {
						if (!(lhs.which() == utree_type::double_type || lhs.which() == utree_type::int_type)
								||
								!(rhs.which() == utree_type::double_type || rhs.which() == utree_type::int_type)
						) {
								return ut;
						}
						if (is_zero_tree(rhs)) {
							return ut;
						}
						else res += (lhs.get<double>() / rhs.get<double>());
					}
					else if (op == "**") {
						if (is_zero_tree(rhs))
							res += 1;
						else {
							if ((!(lhs.which() == utree_type::double_type || lhs.which() == utree_type::int_type))
									||
									!((rhs.which() == utree_type::double_type || rhs.which() == utree_type::int_type)
							))
								return ut;
							if (lhs.get<double>() < 0 && (fabs(rhs.get<double>()) < 1 && fabs(rhs.get<double>()) > 0)) {
								// the result is complex
								// we do not support this (for now)
								return ut;
							}
							res += pow(lhs.get<double>(), rhs.get<double>());
						}
					}
					else if (op == "LN") {
						if (!((lhs.which() == utree_type::double_type) || lhs.which() == utree_type::int_type)) return ut;
						if (lhs.get<double>() > 0) {
							res += log(lhs.get<double>());
						}
						else {
							// outside the domain of ln
							return ut;
						}
					}
					else if (op == "EXP") {
						if (!((lhs.which() == utree_type::double_type) || lhs.which() == utree_type::int_type)) return ut;
						res += exp(lhs.get<double>());
					}
					else return ut;
					op.erase();
				}
				++it;
			}
			if (!is_allowed_value<double>(res)) return ut;
			return utree(res);
			//std::cout << ") ";
			break;
		}
		case utree_type::function_type: {
			break;
		}
		case utree_type::string_type: {
			return ut;
		}
		case utree_type::double_type: {
			double retval = ut.get<double>();
			if (!is_allowed_value<double>(retval)) {
				return ut;
			}
			return utree(retval);
		}
		case utree_type::int_type: {
			double retval = ut.get<double>();
			if (!is_allowed_value<double>(retval)) {
				return ut;
			}
			return utree(retval);
		}
	}
	return ut;
}

// TODO: transitional code for backwards compatibility
boost::spirit::utree const process_utree(
		boost::spirit::utree const& ut,
		evalconditions const& conditions
		) {
	std::map<std::string,int> placeholder;
	double placeholder2[0];
	return process_utree(ut, conditions, placeholder, placeholder2);
}
