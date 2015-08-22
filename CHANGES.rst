What's New
==========

0.2 (2015-08-22)
----------------

This is a big release and is largely incompatible with 0.1.x.
This was necessary for the move to the new equilibrium engine.
0.2.x will be the last "alpha" version of pycalphad where APIs are broken without notice.
0.3 will begin the "beta" cycle where API stability will be enforced.

* pycalphad now depends on numpy>=1.9
* New unified equilibrium computation interface with ``equilibrium`` function.
  Features point, step and map calculation for multi-phase, multi-component problems.
  Time performance is a known issue. A typical calculation will take 3-5 minutes until it's fixed.
* ``...`` can be used in the phases argument of ``equilibrium`` to mean "all phases in a Database".
* ``pycalphad.eq`` is renamed to ``pycalphad.core``
* ``energy_surf`` is now deprecated in favor of the new xray-based ``calculate``
  It's possible to convert xray Datasets to pandas DataFrames with the ``.to_dataframe()`` function.
* The ``Equilibrium`` class has been removed without deprecation. The old engine worked unreliably.
  Use the new ``equilibrium`` routine instead.
* The ``Model`` class has been streamlined. It's now much easier to modify ``Model``s be accessing the
  ``Model.models`` member dict. Changes to ``models`` will be reflected in ``Model.ast``, ``Model.energy``, etc.
* Adding a property attribute to a subclass of ``Model`` automatically makes it available to use in the ``output``
  keyword argument of ``calculate``. This is useful for computing properties not yet defined in ``Model``.
* Experimental support for model parameter fitting is available in the ``residuals`` module.
  It requires the unlisted dependency ``lmfit`` to import.
* BUG: tdb: Sanitize sympify input and clean up pyparsing tracebacks inside parser actions.
* BUG: Always alphabetically sort components listed in interaction parameters (issue #17).
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
