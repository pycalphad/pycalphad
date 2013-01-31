// evaluate.cpp -- evaluate energies from a Database

#include "optimizer.h"
#include <boost/algorithm/clamp.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <limits>
#include <iostream>
#include <fstream>

void evaluate(const Database &DB, const evalconditions &conditions) {
	using namespace boost::numeric::ublas;

	Phase_Collection phase_col;
	// TODO: temporary code for suspending all phases but FCC and liquid
	for (auto i = DB.get_phase_iterator(); i != DB.get_phase_iterator_end(); ++i) {
		if (i->first == "FCC_A1" || i->first == "LIQUID") {
			phase_col[i->first] = i->second;
		}
	}

	const Phase_Collection::const_iterator phase_iter = phase_col.cbegin();
	const Phase_Collection::const_iterator phase_end = phase_col.cend();

	std::ofstream debugfile;
	try {
		// begin reading
		debugfile.open("debug.csv");
		if (!debugfile.good()) BOOST_THROW_EXCEPTION(file_read_error() << boost::errinfo_file_name("debug.csv"));
	}
		catch (file_read_error &e) {
		// 'path' is in scope here, but just for safety we'll read it from the exception object
		std::string fname;
		if (std::string const * mi = boost::get_error_info<boost::errinfo_file_name>(e) ) {
			fname = *mi;
		}
		std::cerr << "Cannot write to \"" << fname << "\"" << std::endl;
	}
	// TODO: check validity of conditions

	// var_map contains indices for all the variable types in our vector
	vector_map var_map;
	// initialize_variables will map out our vector for convenience
	vector<long double> variables = initialize_variables(phase_iter, phase_end, conditions, var_map);
	/*vector<long double> variables (2);
	variables[0] = 0;
	variables[1] = 3;*/
	vector<long double> gradient (variables.size());
	vector<long double> descent_dir (variables.size()); // descent direction vector
	matrix<long double> Hessian = identity_matrix<double>(variables.size()); // init Hessian as identity matrix
	matrix<long double> Hinv = identity_matrix<double>(variables.size()); // inverse of the Hessian
	permutation_matrix<std::size_t> pm (Hessian.size1());
	lu_factorize(Hessian,pm);
	lu_substitute(Hessian,pm, Hinv); // Hinv now contains the inverse of the Hessian matrix

	double step_alpha = 0;

	// TODO: temporary code
	for (auto i = gradient.begin(); i != gradient.end(); ++i) {
		(*i) = 0;
	}
	for (auto i = var_map.sitefrac_iters.begin(); i != var_map.sitefrac_iters.end(); ++i) {
		for (auto j = (*i).begin(); j != (*i).end(); ++j) {
			for (auto k = (*j).begin(); k != (*j).end(); ++k) {
				std::cout << "(" << std::distance(var_map.sitefrac_iters.begin(),i) << ")(" << std::distance((*i).begin(),j) << ")(" << std::distance((*j).begin(),k) << ") is index " << k->second.first << std::endl;
			}
		}
	}

	// Preconditioned Nonlinear Conjugate Gradients with Secant and Polak-Ribiere
	//
	// Construct and update the gradient of the objective function for the initial step
	int iter_index = 0;
	const int iter_max = 100;
	int iter_count = 0;
	const long double epssq = pow(std::numeric_limits<long double>::epsilon(),2);
	update_gradient(phase_iter,phase_end,var_map,variables,gradient,conditions);

	vector<long double> residual = -gradient;
	// TODO: introduce preconditioning if needed
	matrix<long double> precond = identity_matrix<long double>(variables.size());
	vector<long double> precond_residual = prec_prod(precond,residual);
	descent_dir = precond_residual;
	long double delta_new = inner_prod(trans(residual),descent_dir);
	long double delta_0 = delta_new;

	while((iter_index < iter_max) && (delta_new > (epssq*delta_0))) {
		std::cout << "========NEW STEP=======" << std::endl;
		std::cout << "Iteration: " << iter_index << std::endl;
		int alpha_iter = 0;
		const int alpha_iter_max = 500;
		long double delta_d = prec_inner_prod(trans(descent_dir),descent_dir);
		const long double sigma = 1e-6; // TODO: empirical constant; this needs some intelligence
		long double alpha = -sigma;
		vector<long double> next_vars = variables + alpha * descent_dir;
		vector<long double> next_grad (next_vars.size());
		// GRADIENT EVALUATION (next_vars, next_grad)
		update_gradient(phase_iter,phase_end,var_map,next_vars,next_grad,conditions);
		//next_grad[0] = -2*(1 - next_vars[0]) - 400*(next_vars[1] - pow(next_vars[0],2))*next_vars[0];
		//next_grad[1] = 200*(next_vars[1] - pow(next_vars[0],2));
		long double eta_prev = prec_inner_prod(trans(next_grad),descent_dir);
		std::cout << "eta_prev: " << eta_prev << std::endl;

		do {
			// GRADIENT EVALUATION (variables, gradient)
			update_gradient(phase_iter,phase_end,var_map,variables,gradient,conditions);
			//gradient[0] = -2*(1 - variables[0]) - 400*(variables[1] - pow(variables[0],2))*variables[0];
			//gradient[1] = 200*(variables[1] - pow(variables[0],2));
			long double eta = prec_inner_prod(trans(gradient),descent_dir);
			if (abs(eta-eta_prev) < epssq) break; // prevent divide by zero
			std::cout << "eta: " << eta << std::endl;
			std::cout << "eta_prev: " << eta_prev << std::endl;
			alpha = alpha * (eta/(eta_prev - eta));
			std::cout << "alpha: " << alpha << std::endl;
			for (auto i = 0; i < variables.size(); i++) {
				std::cout << "variables[" << i << "] += " << alpha << " * " << descent_dir[i] << std::endl;
				variables[i] += alpha * descent_dir[i];
			}
			eta_prev = eta;
			++alpha_iter;
		}
		while((alpha_iter < alpha_iter_max) && ((pow(alpha,2) * delta_d) > epssq));


		// TODO: Hard enforcement of site fraction constraints
		for (auto i = var_map.sitefrac_iters.begin(); i != var_map.sitefrac_iters.end(); ++i) {
			for (auto j = (*i).begin(); j != (*i).end(); ++j) {
				for (auto k = (*j).begin(); k != (*j).end(); ++k) {
					boost::algorithm::clamp(variables[k->second.first],0,1);
					if (variables[k->second.first] <= 0) variables[k->second.first] = 0;
					if (variables[k->second.first] >= 1) variables[k->second.first] = 1;
					std::cout << "(" << std::distance(var_map.sitefrac_iters.begin(),i) << ")(" << std::distance((*i).begin(),j) << ")(" << std::distance((*j).begin(),k) << ")[" << k->first << "] " << "variables(" << k->second.first << ") = " << variables[k->second.first] << std::endl;
				}
			}
		}

		// TODO: Hard enforcement of phase fraction constraints
		for (auto i = var_map.phasefrac_iters.begin(); i != var_map.phasefrac_iters.end(); ++i) {
			boost::algorithm::clamp(variables[(*i).get<0>()],0,1);
			if (variables[(*i).get<0>()] <= 0) variables[(*i).get<0>()] = 0;
			if (variables[(*i).get<0>()] >= 1) variables[(*i).get<0>()] = 1;
			std::cout << (*i).get<2>()->first << " variables(" << (*i).get<0>() << ") = " << variables[(*i).get<0>()] << std::endl;
		}

		// Calculate the new gradient
		// GRADIENT EVALUATION (variables, gradient)
		update_gradient(phase_iter,phase_end,var_map,variables,gradient,conditions);
		//gradient[0] = -2*(1 - variables[0]) - 400*(variables[1] - pow(variables[0],2))*variables[0];
		//gradient[1] = 200*(variables[1] - pow(variables[0],2));
		residual = -gradient;

		long double delta_old = delta_new;
		long double delta_mid = prec_inner_prod(trans(residual),precond_residual);

		precond = identity_matrix<long double>(variables.size()); // TODO: calculate a preconditioner
		precond_residual = prec_prod(precond,residual);
		
		delta_new = prec_inner_prod(trans(residual),precond_residual);

		long double beta = ((delta_new - delta_mid) / delta_old);
		++iter_count;
		if ((iter_count >= 40) || beta <= 0) {
			descent_dir = precond_residual;
			iter_count = 0;
		}
		else {
			descent_dir = precond_residual + beta * descent_dir;
		}
		++iter_index;



		debugfile << iter_index << ",";
		for (auto i = variables.begin(); i != variables.end(); i++) {
			debugfile << variables[std::distance(variables.begin(),i)] << ",";
			//variables[std::distance(variables.begin(),i)] -= 1e-4 * gradient[std::distance(variables.begin(),i)];
			std::cout << "variables(" << std::distance(variables.begin(),i) << ") = " << variables[std::distance(variables.begin(),i)] << std::endl;
		}
		debugfile << std::endl;
		std::cout << "norm(gradient) = " << norm_2(gradient) << std::endl;
	}

	debugfile.close();
}