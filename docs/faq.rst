.. title:: FAQ

FAQ
===

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


