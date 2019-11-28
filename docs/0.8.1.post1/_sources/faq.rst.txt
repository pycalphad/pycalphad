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

pycalphad does not support parallelization out of the box since version 0.8,
however it is possible to use pycalphad in parallel via packages such as
`dask <http://dask.pydata.org/en/latest/>`_.


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

This bug should now be fixed. Please update to pycalphad 0.7.1 or later.


``RecursionError`` during a calculation
-----------------------------------------

This bug should now be fixed. Please update to pycalphad 0.7.1 or later.


Text is sometimes cut off when saving figures
---------------------------------------------

Occasionally when saving images with the matplotlib function ``plt.savefig``, axis titles and legends are cut off.

This can be fixed:

* Per function call by passing ``bbox_inches='tight'`` keyword argument to ``plt.savefig``
* Locally by running ``import matplotlib as mpl; mpl.rcParams['savefig.bbox'] = 'tight'``
* Permanently by adding ``savefig.bbox : tight`` to your `matplotlibrc file <https://matplotlib.org/users/customizing.html>`_.
