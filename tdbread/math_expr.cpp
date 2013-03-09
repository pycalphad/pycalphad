// math_expr.cpp -- evaluator for FORTRAN-like mathematical expressions

#include "stdafx.h"
#include "warning_disable.h"
#include "conditions.h"
#include <boost/spirit/include/support_utree.hpp>

#include <math.h>
#include <string>

boost::spirit::utree const process_utree(boost::spirit::utree const& ut, evalconditions const& conditions) {
	typedef boost::spirit::utree utree;
	typedef boost::spirit::utree_type utree_type;
	//std::cout << "type: " << ut.which() << std::endl;
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
				if ((*it).which() == utree_type::string_type) { // operator/function
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
						//std::cout << "highlimit:" << highlimit << std::endl;
						if (highlimit == -1) highlimit = curT+1; // highlimit == -1 means no limit
						++it;
						if ((curT >= lowlimit) && (curT < highlimit)) res += process_utree(*it, conditions).get<double>();
						else res += 0;
						//std::cout << "RESULT: " << res << std::endl;
						break;
					}
					++it; // get left-hand side
					// TODO: exception handling
					if (it != end) lhs = process_utree(*it, conditions).get<double>();
					++it; // get right-hand side
					if (it != end) rhs = process_utree(*it, conditions).get<double>();

					if (op == "+") res += (lhs + rhs);  // acculumate the result
					else if (op == "-") {
						if (ut.size() == 2) lhs = -lhs; // case of negation (unary operator)
						res += (lhs - rhs);
					}
					else if (op == "*") res += (lhs * rhs); 
					else if (op == "/") { 
						if (rhs == 0) {
							// TODO: throw a divide by zero exception here
							++it;
						}
						else res += (lhs / rhs);
					}
					else if (op == "**") res += pow(lhs, rhs);
					else if (op == "ln") res += log(lhs);
					else if (op == "exp") res += exp(lhs);
					else {
						// TODO: exception handling
						// a bad symbol made it into our AST
					}
					//std::cout << "LHS: " << lhs << std::endl;
					//std::cout << "RHS: " << rhs << std::endl;
					lhs = rhs = 0;
					op.erase();
				}
				++it;
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
			return ut.get<double>();
			//std::cout << "<double>:" << val;
			break;
		}
		case utree_type::string_type: {
			boost::spirit::utf8_string_range_type rt = ut.get<boost::spirit::utf8_string_range_type>();
			if ((rt.end() - rt.begin()) != 1) {
				// TODO: throw an exception (bad symbol/state variable)
			}
			const char* op(rt.begin());
			if (conditions.statevars.find(*op) != conditions.statevars.end()) {
				//std::cout << "T executed" << std::endl;
				return conditions.statevars.find(*op)->second; // return current value of state variable (T, P, etc)
			}
			else {
				// TODO: throw an exception: undefined state variable
			}
			//std::cout << "<operator>:" << op << std::endl;
			break;
		}
	}
	return utree(utree_type::invalid_type);
}