What's New
==========

0.4.2 (2016-08-26)
------------------

This is a minor feature release with one breaking change.
* There is now support for the Xiong magnetic model (Xiong et al, Calphad, 2012), two-state liquid-amorphous model,
  and Einstein model in the Model class. TDB support has been extended where necessary.
* ENH/BRK: Model: Add 'contributions' class attribute to make it easier for users to define custom energetic
  contributions. The API for custom contributions has changed; the old method will no longer work.
* FIX: equilibrium: Correctly use custom models during property calculation with ``output`` keyword argument.

0.4.1 (2016-08-08)
------------------

This is a minor bug fix release.

* Python 3.3 support has been dropped. See :issue:`46`.
* Documentation has been transitioned to a new domain, [https://pycalphad.org](https://pycalphad.org). See :issue:`47`.
* BLD: Exclude xarray 0.8 from dependencies since it has a regression. (Newer versions are fine.)
* DOC: Automated project documentation building and deployment via Travis CI.

0.4 (2016-08-03)
----------------

This is a major release with bug fixes and performance improvements.

* The equilibrium solver core has been rewritten, resulting in a significant increase in robustness and accuracy,
  particularly for chemical potential calculation with miscibility gaps. See :issue:`43`.
* For performance, dask-powered multiprocessing is now used to parallelize equilibrium calculations.
  Because of this, dask and dill are now dependencies.
* Database and Model objects can now be pickled on all supported platforms, fixing a multiprocessing issue.

0.3.6 (2016-06-01)
------------------

This is a minor release with bug fixes and performance improvements.

* Fix installation problem on Windows when using Anaconda.
* Add new compiled backend for phase models. This new backend provides a significant performance improvement.
* Experimental support for the numba library has been removed.

0.3.5 (2016-05-14)
------------------

This is a minor bug fix release.

* ``tdb``: Fix TDB parsing errors on recent (>=2.1) versions of pyparsing.
* ``equilibrium``: Improve convergence and numerical stability of solver. Fix potential sign error in Hessian matrix.
  Support mapping over two composition variables at once.
  An error is now raised if a calculation specifies components not in the Database.

0.3.4 (2016-04-28)
------------------

This is a minor bug fix release.

* ``Model``: Support the use of the absolute value function in the energy function.

0.3.3 (2016-04-21)
------------------

This is a minor release with bug fixes and performance improvements.

* ``equilibrium``: Significant improvements to the speed and accuracy of the solver.
  There is still some work to do for step and map calculations, planned for 0.4.
* ``Model``: Numerical accuracy improvement for the magnetic model :issue:`40`.
* ``Database``: Improvements to TDB writing, particularly for order-disorder models.
* ``Database``: Support for reading diffusion mobility databases.
  Kinetic simulations are not on the roadmap, but this makes it easier to manipulate diffusion data.
  Pull requests improving pycalphad's support for kinetic calculations are welcome.

0.3.2 (2016-02-22)
------------------

This is a minor bug fix release.

* ``equilibrium``: Fix a bug causing calculations at multiple temperatures to fail in multi-component systems.
  Thanks to Ali for reporting.
* ``equilibrium``: More numerical robustness improvements.
  (Global search now satisfies the strong Wolfe conditions on every iteration.)
  Further performance improvements will come to this soon.
* pycalphad now depends on pyparsing<2.1.0 pending resolution of :issue:`38`.

0.3.1 (2016-02-18)
------------------

This is a minor bug fix release.

* ``Model``: Make the ``curie_temperature`` attribute work when dealing with the order-disorder model.
* ``equilibrium``: Fix a bug involving the ``output`` keyword argument in multi-phase calculations.

0.3 (2016-02-17)
----------------

This is a major release with new features and fixes. It is very likely that
if you will need to update code to be compatible with this version.

* **Breaking change**: Removed ``residuals`` module and the deprecated ``energy_surf`` routine.
* **Breaking change**: Removed ternary isotherm plotting for now, pending a rewrite.
* **Breaking change**: The ``refstates`` module has been renamed to ``refdata``.
* **Breaking change** in ``Database``: Removed ``typedefs`` member.
* ``binplot``:
  Completely rewritten to use the new equilibrium engine. See also the new companion function ``eqplot``.
  **Breaking change**: The API for calling ``binplot`` has also been completely changed.
* ``Database``:
  ``to_file`` learned a ``groupby`` keyword argument for changing how PARAMETERs are sorted.
  Loading a TDB will now raise ``ValueError`` if the file contains duplicate FUNCTIONs.
  The TDB writer now generates output more conformant with Thermo-Calc.
* ``equilibrium``:
  Substantively rewritten for robustness and accuracy. Users will notice a difference, especially for dilute calculations.
  Unfortunately it's still a bit slow; fixing that will be a focus of the 0.3.x cycle. See :issue:`37`.
  Learned a ``output`` keyword argument for specifying additional equilibrium properties to compute.
* The ``tqdm`` library is now a dependency. It adds progress bar support to ``equilibrium``.
* ``Model``:
  Added ``constituents``, ``phase_name`` and ``site_ratios`` attributes, in analogy with ``Phase`` objects.
  This makes it easier to interact with the sublattice model without having to keep ``Database`` objects around.
  Added a ``degree_of_ordering`` (abbreviation ``DOO``) property. Only has meaning for phases with sublattice ordering.
  Added a ``curie_temperature`` (abbreviation ``TC``) property. Only nonzero for phases with magnetic ordering.
* ``calculate``:
  Learned a ``broadcast`` boolean keyword argument for turning broadcasting off. This is useful
  for computing many different system configurations in a pointwise fashion, when there's no
  obvious way of expressing the calculation as a traditional "step" or "map".
* The ``xray`` dependency was renamed to ``xarray``. The change should be transparent to users when updating.

0.2.5 (2015-12-22)
------------------

This is a minor release with new features and bug fixes.

* **Breaking change** in ``Model``: All mixing attributes have been renamed from ``MIX_{attr}`` to ``{attr}_MIX``.
* Early support for reference states has been added to the ``refstates`` module. The reference molar Gibbs energies
  of the pure elements according to the 1991 SGTE standard can be found in ``pycalphad.refstates.SGTE91``.
* ``Database`` now has file import/export support with ``to_file``, ``from_file``, ``from_string`` and ``to_string``.
  Currently TDB is the only supported format, but more can now easily be added in the future.
  The function for extending pycalphad with new formats is ``Database.register_format``.
  Loading databases with the default constructor, i.e., ``Database('file.tdb')``, will continue to work.
* Equivalence comparison support for ``Database`` and ``Model``.
  For example, if ``dbf`` is a ``Database``, ``dbf == Database.from_string(dbf.to_string(fmt='tdb'), fmt='tdb')``.
  Equivalent ``Database`` objects should always produce equivalent ``Model`` objects.
  We have tests for this, but if you find a case where this isn't true, it's a bug and can be reported on the issue tracker.
* A new sampling algorithm for equilibrium calculation, based on the scrambled Halton sequence, has been implemented.
  It should improve performance for multi-component systems once some other improvements have been finalized.
  For now, users will probably not notice a difference.
* ``Model``: Added ``CPM_MIX`` attribute for molar isobaric heat capacity of mixing.
* Many unit tests have been cleaned up and streamlined, with test coverage back up above 80%.

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
