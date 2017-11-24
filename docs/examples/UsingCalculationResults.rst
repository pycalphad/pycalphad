
Exploring ``calculate`` and ``equilibrium`` xarray Datasets
===========================================================

xarray Datasets
---------------

Results returned from calling ``calculate`` or ``equilibrium`` in
pycalphad are `xarray <http://xarray.pydata.org/en/stable/>`__ Datasets.
An xarray Dataset is a data structure that represents N-dimensional
tabular data. It is an N-dimensional analog to the Pandas DataFrame.

This notebook will walk through the structure of xarray Datasets in
pycalphad and some basics of using them. For more in-depth tutorials and
documentation on using xarray Datasets and DataArray's fully, see the
`xarray
documentation <http://xarray.pydata.org/en/stable/index.html>`__.

Dataset structure
-----------------

Each Dataset stores the conditions that properties are calculated at and
the values of the properties as a function of the different conditions.
There are three key terms:

-  ``Dimensions``: these are the conditions that are calculated over,
   e.g. pressure (P) and temperature (T). They are essentially labels.
-  ``Coordinates``: these are the actual *values* that are taken on by
   the dimensions.
-  ``Data variables``: these are the properties calculated by pycalphad,
   such as the Gibbs energy, mixing energy, composition, etc.

``calculate()`` results
-----------------------

Calculate is used to sample properties of a single phase. There are five
dimensions/coordinates:

-  ``P``: pressures (in Pa).
-  ``T``: temperatures (in K).
-  ``component``: the string names of the components in the system
-  ``internal_dof``: The internal\_dof (internal degrees of freedom) is
   the index of the site in any phase's site fraction array. Below the
   FCC\_A1 phase has the sublattice model (AL, ZN) and thus the
   internal\_dof are integers 0 and 1 referring to the AL site (index 0)
   and the ZN site (index 1).
-  ``points``: By default, the calculate function samples points over
   all of the internal degrees of freedom. Each coordinate point simply
   represents the index is a list of all configurations of the
   internal\_dof sampled. There is no underlying physical meaning or
   order.

There are also at least four Data variables:

-  ``Phase``: The string name of the phase. For ``calculate``, this will
   always be the phase name passed.
-  ``X``: The composition of each component in mole fraction as a
   function of the temperature, pressure, and the index of the points
   (there is one composition for each point).
-  ``Y``: The site fraction of each index in the internal\_dof array for
   the given temperature, pressure and point.
-  ``output``: "output" is always whatever property is calculated by the
   output keyword passed to ``calculate``. The default is the molar
   Gibbs energy, GM.

.. code:: ipython3

    %matplotlib inline
    from pycalphad import Database, calculate, equilibrium, variables as v
    
    dbf = Database('alzn_mey.tdb')
    comps = ['AL', 'ZN', 'VA']
    calc_result = calculate(dbf, comps, 'FCC_A1', P=101325, T=[500, 1000])
    print(calc_result)


.. parsed-literal::

    <xarray.Dataset>
    Dimensions:    (P: 1, T: 2, component: 2, internal_dof: 2, points: 4001)
    Coordinates:
      * T          (T) float64 500.0 1e+03
      * component  (component) <U2 'AL' 'ZN'
      * P          (P) float64 1.013e+05
    Dimensions without coordinates: internal_dof, points
    Data variables:
        Y          (P, T, points, internal_dof) float64 1.0 1e-12 1e-12 1.0 ...
        GM         (P, T, points) float64 -1.559e+04 -2.01e+04 -2.01e+04 ...
        X          (P, T, points, component) float64 1.0 1e-12 1e-12 1.0 1e-12 ...
        Phase      (P, T, points) <U6 'FCC_A1' 'FCC_A1' 'FCC_A1' 'FCC_A1' ...


We can manipulate this by selecting data by value (of a coordinate)
using ``sel`` or index (of a coordinate) using ``isel`` similar to a
Pandas array. Below we get the site fraction of ZN (internal\_dof index
of 1 selected by index) at 1000K (selected by value) for the 50th point
(selected by index).

The results of selecting over Data variables gives an xarray DataArray
which is useful for plotting or performing computations on (see
`DataArrays vs
Datasets <http://xarray.pydata.org/en/stable/data-structures.html>`__).

.. code:: ipython3

    print(calc_result.Y.isel(internal_dof=1, points=49).sel(T=1000))


.. parsed-literal::

    <xarray.DataArray 'Y' (P: 1)>
    array([ 0.976488])
    Coordinates:
        T        float64 1e+03
      * P        (P) float64 1.013e+05


accessing the ``values`` attribute on any on any DataArray returns the
multidimensional NumPy array

.. code:: ipython3

    print(calc_result.X.values)


.. parsed-literal::

    [[[[  1.00000000e+00   1.00000000e-12]
       [  1.00000000e-12   1.00000000e+00]
       [  1.00000000e-12   1.00000000e+00]
       ..., 
       [  1.56995650e-01   8.43004350e-01]
       [  1.12072782e-01   8.87927218e-01]
       [  7.43933641e-01   2.56066359e-01]]
    
      [[  1.00000000e+00   1.00000000e-12]
       [  1.00000000e-12   1.00000000e+00]
       [  1.00000000e-12   1.00000000e+00]
       ..., 
       [  1.56995650e-01   8.43004350e-01]
       [  1.12072782e-01   8.87927218e-01]
       [  7.43933641e-01   2.56066359e-01]]]]


``equilibrium()`` results
-------------------------

The Datasets returned by equilibrium are very similar to calculate,
however there are several key differences worth discussing. In
equilibrium Datasets, there are six dimensions/coordinates:

-  ``P``: pressures (in Pa).
-  ``T``: temperatures (in K).
-  ``component``: (Same as calculate) The string names of the components
   in the system.
-  ``internal_dof``: (Same as calculate, except it will be the longest
   possible internal\_dof for all phases) The internal\_dof (internal
   degrees of freedom) is the index of the site in any phase's site
   fraction array. Below the FCC\_A1 phase has the sublattice model (AL,
   ZN) and thus the internal\_dof are integers 0 and 1 referring to the
   AL site (index 0) and the ZN site (index 1).
-  ``X_ZN``: This is the composition of the species that was passed into
   the conditions array. Since we passed ``v.X('ZN')`` to the conditions
   dictionary, this is ``X_ZN``.
-  ``vertex``: The vertex is the index of the phase in equilibrium. The
   vertex has no inherent physical meaning. There will automatically be
   enough to describe the number of phases present in any equilibria
   calculated, implying that vertex can never be large enough to
   invalidate Gibbs phase rule.

There are also at least six Data variables:

-  ``Phase``: The string name of the phase in equilibrium at the
   conditions. There are as many as ``len(vertex)`` phases. Any time
   there are fewer phases in equilibrium than the indices described by
   ``vertex``, the values of phase are paded by ``''``, e.g. for a
   single phase region for FCC\_A1, the values of Phase will be
   ``['FCC_A1', '']``. When more than one phase is present, it is
   important to note that they are not necessarily sorted.
-  ``NP``: Phase fraction of each phase in equilibrium. When there is no
   other equilibrium phase (e.g. single phase ``['FCC_A1', '']``) then
   the value of ``NP`` will be ``nan`` for the absence of a phase,
   rather than 0.
-  ``MU``: The chemical potentials of each component for the conditions
   calculated.
-  ``X``: The equilibrium composition of each element in each phase for
   the calculated conditions.
-  ``Y``: The equilibrium site fraction of each site in each phase for
   the calculated conditions.
-  ``GM``: Same as ``output`` for ``calculate``. It is always reported
   no matter the value of ``output``.
-  ``output``: (optional) "output" is always whatever equilibrium
   property is calculated by the output keyword passed to
   ``equilibrium``. Unlike ``calculate``, this will be in addition to
   the ``GM`` because ``GM`` is always reported.

.. code:: ipython3

    phases = ['LIQUID', 'FCC_A1', 'HCP_A3']
    eq_result = equilibrium(dbf, comps , phases, {v.X('ZN'):(0,1,0.05), v.T: (500, 1000, 100), v.P:101325}, output='HM')
    print(eq_result)


.. parsed-literal::

    <xarray.Dataset>
    Dimensions:    (P: 1, T: 5, X_ZN: 20, component: 2, internal_dof: 2, vertex: 2)
    Coordinates:
      * P          (P) float64 1.013e+05
      * T          (T) float64 500.0 600.0 700.0 800.0 900.0
      * X_ZN       (X_ZN) float64 1e-09 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 ...
      * vertex     (vertex) int64 0 1
      * component  (component) <U2 'AL' 'ZN'
    Dimensions without coordinates: internal_dof
    Data variables:
        X          (P, T, X_ZN, vertex, component) float64 1.0 1e-09 nan nan ...
        GM         (P, T, X_ZN) float64 -1.559e+04 -1.615e+04 -1.649e+04 ...
        Phase      (P, T, X_ZN, vertex) <U6 'FCC_A1' '' 'FCC_A1' '' 'FCC_A1' ...
        NP         (P, T, X_ZN, vertex) float64 1.0 nan 1.0 nan 0.9761 0.02392 ...
        Y          (P, T, X_ZN, vertex, internal_dof) float64 1.0 1e-09 0.9991 ...
        MU         (P, T, X_ZN, component) float64 -1.559e+04 -9.584e+04 ...
        HM         (P, T, X_ZN) float64 5.194e+03 5.859e+03 6.188e+03 6.152e+03 ...
    Attributes:
        engine:   pycalphad 0.5.2.post1+22.gc2a1cf7f.dirty
        created:  2017-11-16T07:56:07.904845


A common operation might be to find the phase fractions of the HCP\_A3
phase as a function of composition for T=800.

However, the only way we can access the values of the phase fraction is
by either the indices or values of the coordinates, we would have to
know which index the HCP\_A3 phase is in before hand to use the ``sel``
or ``isel`` commands.

Since we do not know this, we can do what is called
`masking <http://xarray.pydata.org/en/stable/indexing.html#masking-with-where>`__
to find the data values that match a condition (the Phase is FCC\_A1):

.. code:: ipython3

    print(eq_result.NP.where(eq_result.Phase=='FCC_A1').sel(P=101325, T=800))


.. parsed-literal::

    <xarray.DataArray (X_ZN: 20, vertex: 2)>
    array([[ 1.      ,       nan],
           [ 1.      ,       nan],
           [ 1.      ,       nan],
           [ 1.      ,       nan],
           [ 0.897399,       nan],
           [ 0.71825 ,       nan],
           [ 0.539101,       nan],
           [ 0.359952,       nan],
           [ 0.180803,       nan],
           [ 0.001654,       nan],
           [      nan,       nan],
           [      nan,       nan],
           [      nan,       nan],
           [      nan,       nan],
           [      nan,       nan],
           [      nan,       nan],
           [      nan,       nan],
           [      nan,       nan],
           [      nan,       nan],
           [      nan,       nan]])
    Coordinates:
        P        float64 1.013e+05
        T        float64 800.0
      * X_ZN     (X_ZN) float64 1e-09 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 ...
      * vertex   (vertex) int64 0 1

