pycalphad, a library for the CALculation of PHAse Diagrams
==========================================================

.. image:: https://badges.gitter.im/Join%20Chat.svg
    :target: https://gitter.im/pycalphad/pycalphad
    :alt: Join the chat at https://gitter.im/pycalphad/pycalphad

.. image:: https://coveralls.io/repos/pycalphad/pycalphad/badge.svg?branch=develop&service=github
    :target: https://coveralls.io/github/pycalphad/pycalphad?branch=master
    :alt: Test Coverage

.. image:: https://ci.appveyor.com/api/projects/status/ua1hya8isg588fyp/branch/develop?svg=true
    :target: https://ci.appveyor.com/project/richardotis/pycalphad
    :alt: Windows Build Status

.. image:: https://img.shields.io/travis/pycalphad/pycalphad/master.svg
    :target: https://travis-ci.org/pycalphad/pycalphad
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

* Python 2.7+ or 3.3+
* matplotlib, numpy, scipy, sympy, xarray, pyparsing, tinydb, autograd, tqdm

Installation
------------
See `Installation Instructions`_.

Examples
--------
IPython notebooks with examples are available on `NBViewer`_ and `ReadTheDocs`_.

Documentation
-------------
See the documentation on `ReadTheDocs`_.

Acknowledgements
----------------
Development has been made possible in part through NASA Space Technology Research Fellowship (NSTRF) grant NNX14AL43H, and is supervised by `Prof. Zi-Kui Liu`_ in the `Department of Materials Science and Engineering`_ at the `Pennsylvania State University`_.
We would also like to acknowledge technical assistance on array computations from Denis Lisov.

.. _Installation Instructions: http://pycalphad.readthedocs.org/en/latest/INSTALLING.html
.. _NBViewer: http://nbviewer.ipython.org/github/pycalphad/pycalphad/tree/master/examples/
.. _ReadTheDocs: http://pycalphad.readthedocs.org/
.. _Prof. Zi-Kui Liu: http://www.phases.psu.edu/
.. _Department of Materials Science and Engineering: http://matse.psu.edu/
.. _Pennsylvania State University: http://www.psu.edu/
