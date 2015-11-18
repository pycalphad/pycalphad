What's New
==========

0.2.4 (2015-11-18)
------------------

This is a minor release with bug fixes and performance improvements.

* Optional, experimental support for numba_ has been added to ``calculate``.
  If numba>=0.22 is installed and ``calculate`` is directly called without the `mode`
  keyword argument, a numba-optimized function will be generated for the calculation.
  You can force the old behavior with `mode='numpy'`.
  ``equilibrium`` does not currently use this code path regardless.
* A performance improvement to how ``lower_convex_hull`` computes driving force
  gives a nice speedup when calling ``equilibrium``.
  There's still a lot of room for improvement, especially for step/map calculations.
* Piecewise-defined functions are now lazily-evaluated, meaning only the values necessary
  for the given conditions will be computed. Before, all values were always computed.
  Users will notice the biggest difference when calculating phases with the magnetic model.
* Fix a small but serious bug when running tinydb v3 with pycalphad ( :issue:`30` ).
* Fix a platform-dependent crash bug when using ``binplot`` ( :issue:`31` ).
* Support for numexpr has been removed.
* The documentation on ReadTheDocs should be building properly again ( :issue:`26` ).

.. _numba: http://numba.pydata.org/

0.2.3 (2015-11-08)
------------------

This is a minor release with bug fixes and performance improvements.

* Autograd is now a required dependency. It should be automatically installed on upgrade.
* The magnetic contribution to the energy has been improved in performance.
  For some users (mainly Fe or Ni systems), the difference will be dramatic.
* Numerical stability improvements to the energy minimizer ( :issue:`23` ).
  The minimizer now solves using exact Hessians and is generally more robust.
  `pycalphad.core.equilibrium.MIN_STEP_LENGTH` has been removed.
  There are still issues computing dilute compositions; these will continue to be addressed.
  Please report these numerical issues if you run into them because they are difficult to find through automated testing.
* Automated testing is now enabled for Mac OSX and Windows, as well as Linux (previously enabled).
  This should help to find tricky bugs more quickly. (Note that this runs entirely on separate
  infrastructure and is not collecting information from users.)

0.2.2 (2015-10-17)
------------------

This is a minor bugfix release.

* Numerical stability improvements to the energy minimizer ( :issue:`23` ).
  If you're still getting singular matrix errors occasionally, you can try adjusting
  the value of `pycalphad.core.equilibrium.MIN_STEP_LENGTH` as discussed in the issue above.
  Please report these numerical issues if you run into them because they are difficult to find through automated testing.
* Fixes for the minimizer sometimes giving type conversion errors on numpy 1.10 ( :issue:`24` ).

0.2.1 (2015-09-10)
------------------

This is a minor bugfix release.

* Composition conditions are correctly constructed when the dependent component does not come
  last in alphabetical order ( :issue:`21` ).


0.2 (2015-08-23)
----------------

This is a big release and is largely incompatible with 0.1.x.
This was necessary for the move to the new equilibrium engine.
0.2.x will be the last "alpha" version of pycalphad where APIs are broken without notice.
0.3 will begin the "beta" cycle where API stability will be enforced.

* pycalphad now depends on numpy>=1.9 and xray
* New unified equilibrium computation interface with ``equilibrium`` function.
  Features point, step and map calculation for multi-phase, multi-component problems.
  Time performance is a known issue. A typical calculation will take 3-5 minutes until it's fixed.
* ``Ellipsis`` or ``...`` can be used in the phases argument of ``equilibrium`` to mean "all phases in a Database".
* ``pycalphad.eq`` is renamed to ``pycalphad.core``
* ``energy_surf`` is now deprecated in favor of the new xray-based ``calculate``.
  It's possible to convert xray Datasets to pandas DataFrames with the ``.to_dataframe()`` function.
* The ``Equilibrium`` class has been removed without deprecation. The old engine worked unreliably.
  Use the new ``equilibrium`` routine instead.
* The ``Model`` class has been streamlined. It's now much easier to modify a ``Model`` by accessing the
  ``Model.models`` member dict. Changes to ``models`` will be reflected in ``Model.ast``, ``Model.energy``, etc.
* Adding a property attribute to a subclass of ``Model`` automatically makes it available to use in the ``output``
  keyword argument of ``calculate``. This is useful for computing properties not yet defined in ``Model``.
* Experimental support for model parameter fitting is available in the ``residuals`` module.
  It requires the unlisted dependency ``lmfit`` to import.
* BUG: tdb: Sanitize sympify input and clean up pyparsing tracebacks inside parser actions.
* BUG: Always alphabetically sort components listed in interaction parameters ( :issue:`17` ).
* ENH: V0 TDB parameter support
* ENH: Model: Symbol replacement performance improvement during initialization.
* TST: Test coverage above 80%


0.1.1.post1 (2015-04-10)
------------------------

* Fixes for automated test coverage
* Add funding acknowledgment


0.1.1 (2015-04-09)
------------------

* Single-source version support with Versioneer

0.1 (2015-04-09)
----------------

* Initial public release
