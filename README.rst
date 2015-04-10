pycalphad, a library for the CALculation of PHAse Diagrams
==========================================================

.. image:: https://img.shields.io/coveralls/richardotis/pycalphad.svg
    :target: https://coveralls.io/r/richardotis/pycalphad

.. image:: https://img.shields.io/travis/richardotis/pycalphad/master.svg
    :target: https://travis-ci.org/coagulant/coveralls-python

.. image:: https://pypip.in/version/pycalphad/badge.svg
    :target: https://pypi.python.org/pypi/pycalphad

.. image:: https://pypip.in/py_versions/pycalphad/badge.svg
    :target: https://pypi.python.org/pypi/pycalphad/

.. image:: https://pypip.in/download/pycalphad/badge.svg
    :target: https://pypi.python.org/pypi/pycalphad/

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
by e-mail or on GitHub.

pycalphad is licensed under the MIT License.
See LICENSE.txt for details.

Required Dependencies:
Python 2.7+ or 3.3+ (Python 2.6 is not supported)
Matplotlib, NumPy, SciPy, SymPy, Pandas, PyParsing, TinyDB

Optional Dependencies:
Numexpr (calculation speed-up for multi-core CPUs)

Installation
------------
For the latest stable release, use ``pip install pycalphad``

Examples
--------
IPython notebooks with examples are hosted on NBViewer.
http://nbviewer.ipython.org/github/richardotis/pycalphad/tree/master/examples/

Documentation
-------------
Full documentation is a work in progress. Most routines are documented in
their docstrings, and example code can be found in the 'Examples' section.
