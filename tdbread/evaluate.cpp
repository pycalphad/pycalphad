// evaluate.cpp -- evaluate energies from a Database

#include "database_tdb.h"
#include "optimizer.h"
#include "coin/IpIpoptApplication.hpp"
#include "coin/IpSolveStatistics.hpp"
#include "opt_Gibbs.h"
#include <boost/algorithm/clamp.hpp>
#include <limits>
#include <iostream>
#include <fstream>

using namespace Ipopt;

void evaluate(const Database &DB, const evalconditions &conditions) {
	Phase_Collection phase_col;
	// TODO: temporary code for suspending all phases but FCC and liquid
	for (auto i = DB.get_phase_iterator(); i != DB.get_phase_iterator_end(); ++i) {
		if (i->first == "FCC_A1") {
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

		// Create an instance of your nlp...
		SmartPtr<TNLP> mynlp = new GibbsOpt(phase_iter, phase_end, conditions);

		// Create an instance of the IpoptApplication
		//
		// We are using the factory, since this allows us to compile this
		// example with an Ipopt Windows DLL
		SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

		app->Options()->SetStringValue("derivative_test","first-order");
		app->Options()->SetStringValue("hessian_approximation","limited-memory");
		app->Options()->SetIntegerValue("print_level",12);
		app->Options()->SetStringValue("derivative_test_print_all","yes");

		// Initialize the IpoptApplication and process the options
		ApplicationReturnStatus status;
		status = app->Initialize();
		if (status != Solve_Succeeded) {
			std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
			return;
		}

		status = app->OptimizeTNLP(mynlp);

		if (status == Solve_Succeeded) {
			// Retrieve some statistics about the solve
			Index iter_count = app->Statistics()->IterationCount();
			std::cout << std::endl << std::endl << "*** The problem solved in " << iter_count << " iterations!" << std::endl;

			Number final_obj = app->Statistics()->FinalObjective();
			std::cout << std::endl << std::endl << "*** The final value of the objective function is " << final_obj << '.' << std::endl;
		}
		return;
		// var_map contains indices for all the variable types in our vector
		//vector_map var_map;
		// initialize_variables will map out our vector for convenience
		//vector variables = initialize_variables(phase_iter, phase_end, conditions, var_map);
	/*vector<long double> variables (2);
	variables[0] = 0;
	variables[1] = 3;
	vector gradient (variables.size());
	vector descent_dir (variables.size()); // descent direction vector
	matrix Hessian(variables.size(), variables.size()); // init Hessian as zero matrix
	Hessian.fill(0);

	matrix Hinv = pinv(Hessian); // (psuedo)inverse of the Hessian

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
	// https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
	//
	// Construct and update the gradient of the objective function for the initial step
	int iter_index = 0;
	const int iter_max = 500;
	int iter_count = 0;
	const double epssq = pow(std::numeric_limits<double>::epsilon(),2);
	update_gradient(phase_iter,phase_end,var_map,variables,gradient,conditions);

	vector residual = -gradient;
	// HESSIAN EVALUATION
	calculate_hessian(phase_iter,phase_end,var_map,variables,Hessian,conditions);
	Hessian.print("Hessian: ");
	std::cout << "cond_num: " << norm(Hessian,2) * norm(pinv(Hessian),2) << std::endl;
    Hinv = pinv(Hessian);
	Hinv.print("Hinv: ");
	matrix precond = Hinv;
	std::cout << "assigned Hessian to precond" << std::endl;
	vector precond_residual = precond * residual;
	std::cout << "calculated precond_residual" << std::endl;
	descent_dir = precond_residual;
	double delta_new = dot(residual,descent_dir);
	std::cout << "delta_new: " << delta_new << std::endl;
	double delta_0 = delta_new;

	while((iter_index < iter_max) && (abs(delta_new) > abs(epssq*delta_0))) {
		std::cout << "========NEW STEP=======" << std::endl;
		std::cout << "Iteration: " << iter_index << std::endl;
		int alpha_iter = 0;
		const int alpha_iter_max = 100;
		double delta_d = dot(descent_dir,descent_dir);
		const double sigma = 1e-6; // TODO: empirical constant; this needs some intelligence
	    double alpha = -sigma;
		vector next_vars = variables + alpha * descent_dir;
		vector next_grad (next_vars.size());
		// GRADIENT EVALUATION (next_vars, next_grad)
		update_gradient(phase_iter,phase_end,var_map,next_vars,next_grad,conditions);
		//next_grad[0] = -2*(1 - next_vars[0]) - 400*(next_vars[1] - pow(next_vars[0],2))*next_vars[0];
		//next_grad[1] = 200*(next_vars[1] - pow(next_vars[0],2));
		double eta_prev = dot(next_grad,descent_dir);
		std::cout << "eta_prev: " << eta_prev << std::endl;

		do {
			// GRADIENT EVALUATION (variables, gradient)
			update_gradient(phase_iter,phase_end,var_map,variables,gradient,conditions);
			//gradient[0] = -2*(1 - variables[0]) - 400*(variables[1] - pow(variables[0],2))*variables[0];
			//gradient[1] = 200*(variables[1] - pow(variables[0],2));
			double eta = dot(gradient,descent_dir);
			if (abs(eta-eta_prev) < epssq) break; // prevent divide by zero
			std::cout << "eta: " << eta << std::endl;
			std::cout << "eta_prev: " << eta_prev << std::endl;
			alpha = alpha * (eta/(eta_prev - eta));
			std::cout << "alpha: " << alpha << std::endl;
			for (unsigned int i = 0; i < variables.size(); i++) {
				std::cout << "variables[" << i << "] += " << alpha << " * " << descent_dir[i] << std::endl;
				variables[i] += alpha * descent_dir[i];
			}
			eta_prev = eta;
			++alpha_iter;
		}
		while((alpha_iter < alpha_iter_max) && ((pow(alpha,2) * delta_d) > epssq));


		// Hard enforcement of site fraction constraints
		for (auto i = var_map.sitefrac_iters.begin(); i != var_map.sitefrac_iters.end(); ++i) {
			for (auto j = (*i).begin(); j != (*i).end(); ++j) {
				double sum_subl_frac = 0;
				for (auto k = (*j).begin(); k != (*j).end(); ++k) {
					boost::algorithm::clamp(variables[k->second.first],0,1);
					if (variables[k->second.first] <= 0) variables[k->second.first] = 0;
					if (variables[k->second.first] >= 1) variables[k->second.first] = 1;
					variables[k->second.first+1] = 0;
					variables[k->second.first+2] = 0;
					gradient[k->second.first+1] = 0;
					gradient[k->second.first+2] = 0;
					sum_subl_frac += variables[k->second.first];
					std::cout << std::distance(var_map.sitefrac_iters.begin(),i) << ")(" << std::distance((*i).begin(),j) << ")(" << std::distance((*j).begin(),k) << ")[" << k->first << "] " << "variables(" << k->second.first << ") = " << variables[k->second.first] << std::endl;
				}
				for (auto k = (*j).begin(); k != (*j).end(); ++k) {
					variables[k->second.first] /= sum_subl_frac; // normalize sum of fractions to 1
				}
			}
		}
		// Reset site fraction balance Lagrange multipliers, due to normalization
		for (auto i = var_map.lambda3p_iters.begin(); i != var_map.lambda3p_iters.end(); ++i) {
			variables[i->first] = 0;
			gradient[i->first] = 0;
		}

		// Hard enforcement of phase fraction constraints
		double sum_phase_frac = 0;
		for (auto i = var_map.phasefrac_iters.begin(); i != var_map.phasefrac_iters.end(); ++i) {
			boost::algorithm::clamp(variables[(*i).get<0>()],0,1);
			if (variables[(*i).get<0>()] <= 0) variables[(*i).get<0>()] = 0;
			if (variables[(*i).get<0>()] >= 1) variables[(*i).get<0>()] = 1;
			// Reset Lagrange multipliers for inequality constraint
			variables[(*i).get<0>()+1] = 0;
			variables[(*i).get<0>()+2] = 0;
			gradient[(*i).get<0>()+1] = 0;
			gradient[(*i).get<0>()+2] = 0;
			sum_phase_frac += variables[(*i).get<0>()];
			std::cout << (*i).get<2>()->first << " variables(" << (*i).get<0>() << ") = " << variables[(*i).get<0>()] << std::endl;
		}
		for (auto i = var_map.phasefrac_iters.begin(); i != var_map.phasefrac_iters.end(); ++i) {
			variables[(*i).get<0>()] /= sum_phase_frac; // normalize sum of fractions to 1
		}
		variables[0] = 0; // reset Lagrange multiplier for phase fraction balance
		gradient[0] = 0;

		// Calculate the new gradient
		// GRADIENT EVALUATION (variables, gradient)
		update_gradient(phase_iter,phase_end,var_map,variables,gradient,conditions);
		//gradient[0] = -2*(1 - variables[0]) - 400*(variables[1] - pow(variables[0],2))*variables[0];
		//gradient[1] = 200*(variables[1] - pow(variables[0],2));
		residual = -gradient;

		double delta_old = delta_new;
		double delta_mid = dot(residual,precond_residual);

		calculate_hessian(phase_iter,phase_end,var_map,variables,Hessian,conditions);
		std::cout << "HESSIAN:" << std::endl;
		std::cout << Hessian << std::endl;
		Hinv = pinv(Hessian);
		std::cout << "HINV:" << std::endl;
		std::cout << Hinv << std::endl;
		precond = Hinv;
		precond_residual = precond * residual;
		
		delta_new = dot(residual,precond_residual);

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
		std::cout << "norm(gradient) = " << norm(gradient, 2) << std::endl;
	}
	*/
	debugfile.close();
}