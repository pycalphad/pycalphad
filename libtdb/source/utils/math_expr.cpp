// math_expr.cpp -- evaluator for FORTRAN-like mathematical expressions

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/warning_disable.hpp"
#include "libtdb/include/conditions.hpp"
#include "libtdb/include/exceptions.hpp"
#include <boost/spirit/include/support_utree.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <math.h>
#include <string>

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

boost::spirit::utree const process_utree(boost::spirit::utree const& ut, evalconditions const& conditions) {
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
						BOOST_THROW_EXCEPTION(floating_point_error() << str_errinfo("Calculated value is infinite, subnormal, or not a number"));
					}
					return retval;
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
						++it;
						if ((curT >= lowlimit) && (curT < highlimit)) {
							// Range check satisfied
							// Process the tree and return the result
							// Note: by design we only return the first result to satisfy the criterion
							return process_utree(*it, conditions).get<double>();
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
							return conditions.statevars.find(*lonesym)->second; // return current value of state variable (T, P, etc)
						}
					}
					++it; // get left-hand side
					// TODO: exception handling
					if (it != end) lhs = process_utree(*it, conditions).get<double>();
					++it; // get right-hand side
					if (it != end) rhs = process_utree(*it, conditions).get<double>();

					if (op == "+") res += (lhs + rhs);  // accumulate the result
					else if (op == "-") {
						if (ut.size() == 2) lhs = -lhs; // case of negation (unary operator)
						res += (lhs - rhs);
					}
					else if (op == "*") res += (lhs * rhs); 
					else if (op == "/") { 
						if (rhs == 0) {
							BOOST_THROW_EXCEPTION(divide_by_zero_error());
						}
						else res += (lhs / rhs);
					}
					else if (op == "**") {
						if (lhs < 0 && (abs(rhs) < 1 && abs(rhs) > 0)) {
							// the result is complex
							// we do not support this (for now)
							BOOST_THROW_EXCEPTION(domain_error() << str_errinfo("Calculated values are not real"));
						}
						res += pow(lhs, rhs);
					}
					else if (op == "ln") {
						if (lhs <= 0) {
							// outside the domain of ln
							BOOST_THROW_EXCEPTION(domain_error() << str_errinfo("Logarithm of nonpositive number is not defined"));
						}
						res += log(lhs);
					}
					else if (op == "exp") res += exp(lhs);
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
			double retval = ut.get<double>();
			if (!is_allowed_value<double>(retval)) {
				BOOST_THROW_EXCEPTION(floating_point_error() << str_errinfo("Calculated value is infinite, subnormal, or not a number"));
			}
			return retval;
		}
		case utree_type::string_type: {
			boost::spirit::utf8_string_range_type rt = ut.get<boost::spirit::utf8_string_range_type>();
			const char* op(rt.begin());
			if ((rt.end() - rt.begin()) != 1) {
				// throw an exception (bad symbol/state variable)
				BOOST_THROW_EXCEPTION(bad_symbol_error() << str_errinfo("Non-arithmetic (e.g., @) operators or state variables can only be a single character") << specific_errinfo(op));
			}
			if (conditions.statevars.find(*op) != conditions.statevars.end()) {
				//std::cout << "T executed" << std::endl;
				return conditions.statevars.find(*op)->second; // return current value of state variable (T, P, etc)
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
