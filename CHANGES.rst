What's New
==========

0.9.1 (2021-08-08)
------------------

This is a minor release containing performance improvements and bug fixes.

* ENH: Add metastable composition sets to solver starting point ( :issue:`362` )
* ENH: Refactor solver and improve solver performance ( :issue:`357`, :issue:`360` )
* FIX: Custom state variables cannot be set equal to zero ( :issue:`365` )
* ENH: Allow passing dictionaries of PhaseRecord objects to equilibrium and calculate ( :issue:`361` )
* FIX: Database parsing fails if some tokens are not uppercase ( :issue:`364` )
* ENH: Add parsing support for all TC parameters as of 2021b documentation ( :issue:`358` )


0.9.0 (2021-06-06)
------------------

This is a major release including a new minimizer, support for installing from PyPI using pip, performance improvements, documentation improvements, and bug fixes.

* ENH: Introduce a new energy minimizer based on the method described by [Sundman2015]_.
  The new minimizer improves performance, convergence for dilute and pseudo-binary systems,
  and reduces the point grid density (memory) required for convergence.
  ( :issue:`329`, :issue:`332`, :issue:`336`, :issue:`339`, :issue:`343`, :issue:`344` )
* BLD: Improve the build system to support PEP 517 and PEP 518 isolated builds and
  the ``pyproject.toml`` standard. ( :issue:`330`, :issue:`331`, :issue:`334` )
* BLD: Distributed pycalphad wheels on PyPI in addition to the conda-forge package.
  Using pip to install pycalphad is now supported and recommended. ( :issue:`346` )
* ENH: Improve performance of ``Model`` instantiation ( :issue:`340` )
* MAINT: Add support for pyparsing 3.0 ( :issue:`348` )
* DOC: Improve installation documentation with the newly supported pip/PyPI packages.
  An installation configuration tool is provided in the installation guide. ( :issue:`351` )
* MAINT: Refactor _sample_phase_constitution ( :issue:`335` )
* MAINT: Remove ``refdata.py`` that was deprecated in pycalphad 0.6 ( :issue:`333` )
* MAINT: Using setuptools_scm instead of versioneer to version pycalphad.
  The version scheme for development versions has changed. ( :issue:`341` )


0.8.5 (2021-05-20)
------------------

* MAINT: Introduce a warning when partitioned phase models incorrectly use ``_MIX`` properties ( :issue:`328` )
* FIX: Generalize assumptions for the species that can be in substitutional and interstitial sublattices of partitioned phase models ( :issue:`311` )
* FIX: Improve strictness when parsing TDB FUNCTION and PARAMETER lines ( :issue:`308` )
* FIX: Fix Triangular axes projections to allow padding for text labels ( :issue:`295` )
* ENH: Phase diagram plotting: enable tie-line/triangle and legend customization ( :issue:`292` )
* FIX: Fix a bug where ``Model._array_validity`` could include species that are not in the phase constituents ( :issue:`258` )
* FIX: Fix but where ``eqplot`` could attempt to plot tie-triangles for T-X diagrams ( :issue:`288` )

* MAINT: Dependency and build system changes:

  * Support Python 3.9 ( :issue:`298` )
  * Bump cyipopt to support new package name and v1.0 API ( :issue:`289` )
  * Bump SymPy pin to v1.8 ( :issue:`289` )
  * Bump SymEngine pin to v0.7.0 ( :issue:`316` )
  * Bump SymEngine.py pin to v0.7.2 ( :issue:`289` )
  * Switch to matplotlib-base; bump to v3.3 ( :issue:`327` )
  * Fix NumPy deprecation warnings introduced in v1.20 ( :issue:`312` )
  * Use `setup_requires` for build Python build dependences ( :issue:`325` )
  * The pycalphad conda channel is no longer required for installation ( :issue:`297` )


0.8.4 (2020-10-28)
------------------

This is a minor release containing performance improvements and bug fixes.

* DOC: Fix array indexing in examples ( :issue:`282` )
* ENH: Improve compilation performance by unwraping ``Piecewise`` with only one nonzero branch in ``Model.redlich_kister_sum`` ( :issue:`281` )
* ENH: Improve ``hyperplane()`` performance and support parameter vectorization in ``calculate()`` ( :issue:`274` )
* FIX: Bug fixes and tests for the two sublattice ionic liquid where energies were calculated incorrectly ( :issue:`273` )
* MAINT: Fixes an internal API regression in ``_eqcalculate``, the ``models`` aregument is now ``model`` ( :issue:`272` )
* FIX: Fixes a bug where databases with many components would raise an error because ``_eqcalculate`` computed the degrees of freedom based on  all components instead of the active components defined in the current ``Model`` instance ( :issue:`270` )

0.8.3 (2020-03-31)
------------------

This is a minor bug fix release.

* FIX: Improved ``model_hints`` construction when reading databases with out of order type definitions, fixes detecting disordered phases with ``filter_phases`` ( :issue:`269` )
* FIX: Complex infinity in ``Model`` expressions are converted to real infinity so SymEngine can ``lambdify`` the expressions ( :issue:`267` )

0.8.2 (2020-03-07)
------------------

This is a minor release with bug fixes and performance improvements. Python 2.7 support is dropped as well as Python 3.5 and below. Python 3.6-3.8 are explictly supported.

* ENH: Reading large databases via delayed parameter processing ( :issue:`266` )
* FIX: Support PhaseRecord pickling, switch SymEngine backend to LLVM ( :issue:`264` )
* DOC: Regenerate examples ( :issue:`263` )
* DOC: Update examples ( :issue:`262` )
* ENH: variables.MassFraction object implementation ( :issue:`254` )
* MAINT: Update and pin to SymPy 1.5 ( :issue:`251` )
* MAINT: Support Python 3.8, drop Python 2, <3.6 ( :issue:`257` )

0.8.1 (2019-11-28)
------------------

This is a minor release with bug fixes and performance improvements.

* ENH: Calculation speed and accuracy improvements via exact Hessians and the SymEngine lambda backend ( :issue:`249` )
* ENH: Faster binary phase diagram mapping ( :issue:`209` )
* FIX: Calculating disordered phase only if respective ordered phase inactive. Thanks @igorjrd ( :issue:`248` )
* ENH: Use better colors in phase_legend(). Thanks @igorjrd ( :issue: `242` )
* FIX: Suspend a phase if only a pure-vacancy endmember would be active. Thanks @igorjrd ( :issue:`239` )
* ENH: Add element reference data reading/writing to TDB parser ( :issue:`240` )
* DOC: Typo in documentation. Thanks @jwsiegel2150 ( :issue:`237` )
* FIX: SymPy namespace clash with TDBs, and other deprecation fixes ( :issue:`234` )
* DOC: Update installation instructions ( :issue:`241` )
* MNT: Relax dask requirements to the minimum required for `scheduler=` syntax ( :issue:`223` )

0.8 (2019-05-31)
----------------

This is a major release with bug fixes and performance improvements.

* ENH: Major performance improvement with new Just-In-Time SymEngine/LLVM-based compiler. ( :issue:`220` )
* ENH: Support for fixing the chemical potential of an element as an equilibrium constraint. ( :issue:`200` )
* ENH: Support for shifting the reference state of an equilibrium calculation. ( :issue:`205` )
* MAINT: Internal reorganization of the phase model constructors. ( :issue:`214` :issue:`217` )
* DOC: A new example for computing properties of custom models has been added.
* MAINT: Windows Python 2.7 support has been dropped. ( :issue:`220` )


0.7.1 (2018-11-14)
------------------

This is a minor release with bug fixes and performance improvements.

* FIX: PhaseRecord: Fix pickling, so distributed scheduling will work ( :issue:`196` )
* FIX: Max phases by Gibbs phase rule accommodated  ( :issue:`184` )
* FIX: SymPy 1.2 compatibility ( :issue:`180` )
* FIX: Model: Degree of ordering property calculation when vacancy is in the system
* FIX: Species Python 2 unicode support ( :issue:`166` )
* ENH: Allow solution refinement by the Ipopt solver to be disabled. ( :issue:`187` )
* ENH: Enable custom solvers ( :issue:`177` )
* DOC: Update pycalphad logo to be smoother and have a version with text. Thanks to Joyce Yong. ( :issue:`193` )
* MNT: Refactor callables creation in equilibrium() and calculate() ( :issue:`192` )
* ENH: tdb: Move tdb grammar creation out of loop
* ENH: Add magnetic moment as default Model property BMAG
* ENH: Optimize _compute_phase_values ( :issue:`175` )


0.7 (2018-03-19)
----------------

This is a major release with new features and performance improvements.

* ENH: Add support for calculations with species, including support for the associate, ionic liquid, and gas phase models ( :issue:`161` ).
* The compiled backed of common models has been removed. Users should expect that the first set of calculations with new phases in a Python script or session be slower as the models for each phase are compiled in real time.
* ENH: Performance of JIT compilation of phases has been improved.
* ENH: equilibrium: Performance optimizations to reduce the overhead of calling equilibrium, particularly in tight loops.


0.6.1 (2017-12-01)
------------------

This is a minor release with bug fixes and new features.

* ENH: tdb: Add more command parsing: TEMPERATURE_LIMITS, DATABASE_INFO, VERSION_DATE, REFERENCE_FILE, ADD_REFERENCES
* FIX: tdb: Allow '-' character in phase names.
* ENH/FIX: tdb: Allow comma character to specify default low temperature limit (0.01 K)


0.6 (2017-11-26)
----------------

This is a major release with new features, bug fixes and performance improvements.

* Users updating from an earlier version should follow the updated installation instructions to ensure they have all the correct dependencies.
* MAINT: Python 3.4 support has been dropped ( :issue:`145` ).
* MAINT: Windows Python 2.7 32-bit support has been dropped. 64-bit is still supported.
* ENH: A new solver based on the optimization package IPOPT has been implemented, leading to increased accuracy and lower memory consumption ( :issue:`124` ).
* ENH: Windows users no longer have to install the Microsoft C compiler if they use Anaconda. The installer will now automatically download a MinGW-based compiler toolchain.
* DOC: The documentation has been updated and expanded ( :issue:`146` ).
* ENH: calculate: Automatically suspend inactive phases from calculation ( :issue:`141` ).
* ENH: Tielines can now be toggled on and off in phase diagrams ( :issue:`136` ).
* ENH: Species support in Database and TDB read/write ( :issue:`137` ).
* FIX: Axis labeling bug in eqplot due to leaking list comprehension variable.
* FIX: Maintain sorted state variable ordering when one or more state variables is left as default ( :issue:`116` ).
* MAINT: Cleanup refdata, fitting, and core.eqresult modules ( :issue:`135` ).
* FIX: tdb: Update float parsing regex ( :issue:`144` ).


0.5.2 (2017-08-10)
------------------

This is a minor release with a new feature, bug fixes and performance improvements.

* ENH: Add ternary isothermal phase diagram plotting. ( :issue:`98` ).
* FIX: sympy 1.1 compatibility ( :issue:`108` ).
* ENH/FIX: Make equilibrium Datasets serializable to netCDF ( :issue:`111` ).
* FIX: Raise an error if invalid keyword arguments are passed to Database.write ( :issue:`117` ).
* ENH/DOC: Remove log.py module ( :issue:`104` ).
* FIX: Mistake in the Cementite Analysis example ( :issue:`91` ).


0.5.1 (2017-05-12)
------------------

This is a minor release with bug fixes.

* FIX: Custom Models involving certain mathematical constants will compile. Fixes :issue:`91`.
* FIX: Undefined symbols in CompiledModel are automatically set to zero. Fixes :issue:`90`.

0.5 (2017-05-04)
----------------

This is a major release with bug fixes and performance improvements.

* Python 3.6 is now supported. Python 3.3 support has been dropped.
* The equilibrium solver is now significantly faster and more robust. A new Cython-based implementation of the Model class,
  CompiledModel, has virtually eliminated cold-start calculation time.
* Cython is now a run-time and build-time dependency. Obsolete dependencies have been removed. Windows is still supported
  with the caveat that users will need to install the Microsoft Visual C++ Build Tools to get a working C compiler.
* The [pycalphad paper](http://doi.org/10.5334/jors.140) has been published.
* The progress bar has been removed along with the dependency on tqdm.
* ENH: Raise warning if unused kwargs are passed to equilibrium
* ENH: TDB compatibility: All characters after command delimiters should be ignored.
* FIX: Fix solver when sum of compositions > 1
* DOC: calculate: Add default pdens value to docstring. Fixes  :issue:`85`.
* FIX: Indexing errors ( :issue:`63` ).
* FIX: eqsolver: Handle component index correctly when VA is not last component in alphabetical order. Fixes :issue:`62`.
* ENH: calculate/equilibrium: Add parameters kwarg to allow users to override Database FUNCTIONs.
* DOC: Add Getting Help section to readme and docs.
* FIX: binplot: Fix ordering of phase labels and colors.
* tdb: Make ELEMENT grammar more strict to catch typos easier. Fixes :issue:`57`.
* ENH: Caching rewrite and performance increase. Database objects are now hashable.
* ENH: calculate: Performance enhancements via profiling.
* ENH: equilibrium: Break computation up into parallelizable pieces using dask.

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

.. [Sundman2015] Sundman, Lu, and Ohtani, *Computational Materials Science* 101 (2015) 127-137 `doi: 10.1016/j.commatsci.2015.01.029 <http://doi.org/10.1016/j.commatsci.2015.01.029>`_
