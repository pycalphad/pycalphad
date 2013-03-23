/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// equilibrium.hpp -- declaration for Equilibrium object

/*
 * What this class needs to do:
 * This class will be the foundational piece of how TDBread interacts with
 * libGibbs. Code will construct a Database (using libTDB) and set conditions
 * using an evalconditions object. They will then pass those parameters to an
 * Equilibrium object. The ctor will minimize the Gibbs energy of the system
 * and find the hypersurface corresponding to the n-phase equilibrium.
 * I will move a lot of the evaluate.cpp code into the Equilibrium object.
 * Once minimization is complete, I will have the values of the phase fractions
 * and sublattice site fractions which minimize the Gibbs energy.
 * For convenience in debugging, I will overload an operator to make Equilibrium
 * objects prettyprint in a Thermo-Calc style way.
 * Exception handling will have to be done carefully since a lot of numerical
 * stuff will be happening in the ctor.
 *
 * Equilibrium objects will contain:
 * 1) the name of the database used (for compatibility checking)
 * 2) the value of the minimum Gibbs energy (J/mol)
 * 3) the evalconditions object used for minimization
 * 3) For each phase:
 * 		a) name of phase
 * 		b) fraction of phase
 * 		c) For each sublattice: the stoichiometric coefficient
 * 		d) For each sublattice: the site fraction of each species
 *
 * Equilibrium objects will have:
 * 1) Gibbs minimization for m species and n phases
 * 2) Convenience function for converting site fraction to mole fraction (phase and overall)
 * 3) prettyprint functionality by overloading insertion operator
 * 4) getter functions for the conditions of the equilibrium
 * 4) (FUTURE) the ability to be constructed from arbitrary (e.g. experimental) data
 */
class Equilibrium {

};
