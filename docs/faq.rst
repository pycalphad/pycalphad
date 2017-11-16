.. title:: FAQ

FAQ
===

What units does pycalphad use?
------------------------------

* All units are SI units.
* Molar quantities are used for state variables, e.g. energy has units ``J/mol``.
* Composition and site occupancies are mole fractions.


Is any parallelism supported in pycalphad?
------------------------------------------

Equilibrium calculations in pycalphad can be parallelized using `dask <http://dask.pydata.org/en/latest/>`_ out of the box.
Several schedules are supported in `dask <http://dask.pydata.org/en/latest/scheduler-overview.html>`_
and some have been `benchmarked in pycalphad <https://github.com/pycalphad/pycalphad/issues/101>`_,
where the ``Client`` scheduler was found to be give a mild performance boost.

The ``Client`` scheduler can be used as in an equilibrium calculation as follows

.. code-block:: python

    from distributed import LocalCluster, Client
    from pycalphad import equilibrium, Database, variables as v

    # Will parallelize calculations over all available cores by default
    # See the LocalCluster API for more options:
    # https://distributed.readthedocs.io/en/latest/local-cluster.html
    lc = LocalCluster()
    scheduler = Client(lc)

    # set up and run the equilibrium calculation using the Client scheduler
    dbf = Database('Ti-V.tdb')
    comps = ['TI', 'V', 'VA']
    phases = ['BCC_A2', 'HCP_A3', 'LIQUID']
    conditions = {v.P: 101325, v.T: 300, v.X('V'): (0, 1, 0.01)}

    eq = equilibrium(dbf, comps, phases, conditions, scheduler=scheduler)


How long should equilibrium calculations take?
----------------------------------------------

Roughly speaking, single point equilibrium calculations should take on the order
of 200ms.

The ``binplot`` and ``ternplot`` functions construct phase diagrams by
a dense grid of point calculations over the conditions passed. The phase diagrams
are mapped by the tieline points of the two phase regions, so unless there are
two phase regions in a very small composition range, only coarse composition
grids are required for phase diagram calculations.


``TypeError: argument is not an mpz`` during a calculation
----------------------------------------------------------

This is an upstream bug in sympy, where floats are unable to be pickled.
The fix has been completed, but not yet released. While the fix is not released,
removing the gmpy2 package from their Python environment (e.g.
``conda remove --force gmpy2``) will fix the error. Alternatively, setting the
environment variable ``MPMATH_NOGMPY`` to a non-zero value will fix the error.


Text is sometimes cut off when saving figures
---------------------------------------------

Occasionally when saving images with the matplotlib function ``plt.savefig``, axis titles and legends are cut off.

This can be fixed:

* Per function call by passing ``bbox_inches='tight'`` keyword argument to ``plt.savefig``
* Locally by running ``import matplotlib as mpl; mpl.rcParams['savefig.bbox'] = 'tight'``
* Permanently by adding ``savefig.bbox : tight`` to your `matplotlibrc file <https://matplotlib.org/users/customizing.html>`_.
