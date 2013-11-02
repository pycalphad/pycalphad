/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// definition for EquilibriumFactory object

#include "libgibbs/include/libgibbs_pch.hpp"
#include "libgibbs/include/equilibrium.hpp"
#include "external/coin/IpIpoptApplication.hpp"
#include "external/coin/IpSolveStatistics.hpp"
#include <boost/make_shared.hpp>

using namespace Ipopt;


EquilibriumFactory::EquilibriumFactory() : app(IpoptApplicationFactory()) {
	// set Ipopt options
	app->Options()->SetStringValue("derivative_test","second-order");
	app->Options()->SetNumericValue("derivative_test_perturbation",1e-6);
	//app->Options()->SetStringValue("hessian_approximation","limited-memory");
	app->Options()->SetIntegerValue("print_level",4);
	app->Options()->SetStringValue("derivative_test_print_all","yes");
	app->Options()->SetStringValue("sb","yes"); // we handle copyright printing for Ipopt


	ApplicationReturnStatus status;
	status = app->Initialize();
	if (status != Solve_Succeeded) {
		BOOST_THROW_EXCEPTION(equilibrium_error() << str_errinfo("Error initializing solver"));
	}
}

boost::shared_ptr<Equilibrium> EquilibriumFactory::create
(const Database &DB, const evalconditions &conds) {
	return boost::make_shared<Equilibrium>(DB, conds, app);
}

boost::shared_ptr<IpoptApplication> EquilibriumFactory::GetIpopt() { return app; }
