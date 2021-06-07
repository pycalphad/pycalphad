pycalphad, a library for the CALculation of PHAse Diagrams
==========================================================

.. image:: https://badges.gitter.im/Join%20Chat.svg
    :target: https://gitter.im/pycalphad/pycalphad
    :alt: Join the chat at https://gitter.im/pycalphad/pycalphad

.. image:: https://codecov.io/gh/pycalphad/pycalphad/branch/develop/graph/badge.svg?token=Fu7FJZeJu0
    :target: https://codecov.io/gh/pycalphad/pycalphad
    :alt: Test Coverage

.. image:: https://github.com/pycalphad/pycalphad/workflows/Tests/badge.svg
    :target: https://github.com/pycalphad/pycalphad/actions?query=workflow%3ATests
    :alt: Build Status

.. image:: https://img.shields.io/pypi/status/pycalphad.svg
    :target: https://pypi.python.org/pypi/pycalphad/
    :alt: Development Status

.. image:: https://img.shields.io/pypi/v/pycalphad.svg
    :target: https://pypi.python.org/pypi/pycalphad/
    :alt: Latest version

.. image:: https://img.shields.io/pypi/pyversions/pycalphad.svg
    :target: https://pypi.python.org/pypi/pycalphad/
    :alt: Supported Python versions

.. image:: https://img.shields.io/pypi/l/pycalphad.svg
    :target: https://pypi.python.org/pypi/pycalphad/
    :alt: License

**Note**: Unsolicited pull requests are _happily_ accepted!

pycalphad is a free and open-source Python library for
designing thermodynamic models, calculating phase diagrams and
investigating phase equilibria within the CALPHAD method. It
provides routines for reading Thermo-Calc TDB files and for
solving the multi-component, multi-phase Gibbs energy
minimization problem.

The purpose of this project is to provide any interested people
the ability to tinker with and improve the nuts and bolts of
CALPHAD modeling without having to be a computer scientist or
expert programmer.

For assistance in setting up your Python environment and/or
collaboration opportunities, please contact the author
by e-mail or using the issue tracker on GitHub.

pycalphad is licensed under the MIT License.
See LICENSE.txt for details.

Required Dependencies:

* Python 3.7+
* matplotlib, numpy, scipy, sympy, symengine, xarray, pyparsing, tinydb

Installation
------------
See `Installation Instructions`_.

Examples
--------
Jupyter notebooks with examples are available on `NBViewer`_ and `pycalphad.org`_.

Documentation
-------------
See the documentation on `pycalphad.org`_.

Getting Help
------------

Questions about installing and using pycalphad can be addressed in the `pycalphad Google Group`_.
Technical issues and bugs should be reported on on `GitHub`_.
A public chat channel is available on `Gitter`_.

.. _Gitter: https://gitter.im/pycalphad/pycalphad
.. _GitHub: https://github.com/pycalphad/pycalphad
.. _pycalphad Google Group: https://groups.google.com/d/forum/pycalphad

Citing
------

If you use pycalphad in your research, please consider citing the following work:

Otis, R. & Liu, Z.-K., (2017). pycalphad: CALPHAD-based Computational Thermodynamics in Python. Journal of Open Research Software. 5(1), p.1. DOI: http://doi.org/10.5334/jors.140

Acknowledgements
----------------
Development has been made possible in part through NASA Space Technology Research Fellowship (NSTRF) grant NNX14AL43H, and is supervised by `Prof. Zi-Kui Liu`_ in the `Department of Materials Science and Engineering`_ at the `Pennsylvania State University`_.
We would also like to acknowledge technical assistance on array computations from Denis Lisov.

.. _Installation Instructions: http://pycalphad.org/docs/latest/INSTALLING.html
.. _NBViewer: http://nbviewer.ipython.org/github/pycalphad/pycalphad/tree/master/examples/
.. _pycalphad.org: http://pycalphad.org/
.. _Prof. Zi-Kui Liu: http://www.phases.psu.edu/
.. _Department of Materials Science and Engineering: http://matse.psu.edu/
.. _Pennsylvania State University: http://www.psu.edu/
