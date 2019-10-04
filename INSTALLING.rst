Installation Instructions
=========================

Pycalphad is best installed by using Anaconda_. Release versions of pycalphad
are available on PyPI, however pycalphad depends on the externally developed
`Ipopt`_ and `SymEngine`_ libraries for numerical optimization and symbolic
calculations and must be installed separately in order to install pycalphad
from PyPI. For this reason, it is *strongly* recommended to use the Anaconda
Python distribution to install pycalphad, rather than using ``pip``.

Anaconda
--------

For all Windows, macOS and Linux platforms, it is recommended to use Anaconda_
to install the latest release of pycalphad. Anaconda is a scientific Python
distribution by Anaconda, Inc. It provides good support for various scientific
packages and otherwise challenging to install packages.

To install pycalphad from Anaconda

1. Download and install Anaconda_
2. From the Anaconda Prompt (Windows) or a terminal emulator (macOS and Linux) run ``conda install -c pycalphad -c conda-forge pycalphad``


Development Versions (Advanced Users)
-------------------------------------

Installing a development version of pycalphad will allow the latest changes and
features in pycalphad to be used by running the bleeding edge version of the
code, even if those changes have not been released in a tagged version to PyPI
or Anaconda.

Running the development versions allows pycalphad to be run directly from the
modifiable source code, so you can add new features and contribute them back to
the project.

Installing
~~~~~~~~~~

To install the latest development version of pycalphad, run from the Anaconda
Prompt (Windows) or a terminal emulator (macOS or Linux):

1. Install pycalphad and the conda-forge C and C++ compilers ``conda create -c pycalphad -c conda-forge c-compiler cxx-compiler pycalphad``
#. Remove the installed pycalphad package so the development version can be installed ``conda remove --force pycalphad``
#. Get the pycalphad source ``git clone https://github.com/pycalphad/pycalphad.git pycalphad/`` (or download from https://github.com/pycalphad/pycalphad)
#. Go to the top level directory of the package ``cd pycalphad``
#. Run ``pip install -e .``

Updating
~~~~~~~~

By default, the development version installed will track the latest changes in
the ``develop`` branch of pycalphad, tracked on
`GitHub <https://github.com/pycalphad/pycalphad/tree/develop>`_.
Two steps need to be performed to update the code: modifying/updating the source
code and compiling any changes in the Cython files.

To update the code in the current branch to the latest changes on GitHub, the
changes are pulled from GitHub by running ``git pull`` from the Anaconda Prompt
or terminal emulator.

To switch to a different branch, e.g. ``master`` (which tracks the latest
released version) or another feature branch, run ``git checkout <branch>``,
where ``<branch>`` is the name of the branch to checkout, such as ``master``
(without the brackets ``<`` and ``>``).

The Cython files can be recompiled by running ``python setup.py build_ext --inplace``
from the top level ``pycalphad`` directory which contains the ``setup.py`` file.

Troubleshooting
---------------

During installation
~~~~~~~~~~~~~~~~~~~

``IpStdCInterface.h: No such file or directory``
++++++++++++++++++++++++++++++++++++++++++++++++

During installation via pip, the error
``src/cyipopt.c:239:29: fatal error: IpStdCInterface.h: No such file or directory``
indicates that the headers for Ipopt library cannot be found when trying to install
cyipopt.

The library Ipopt is a new dependency in pycalphad 0.6 and requires installation
of an external library not available on PyPI. This error message means Ipopt is
either not installed or not in your PATH. Ipopt installation instructions are
available at https://www.coin-or.org/Ipopt/documentation/node10.html.

To continue installation via pip:

1. Install Ipopt
2. Download the cyipopt package source from https://github.com/matthias-k/cyipopt
3. Install the downloaded cyipopt package. You may need to add the directories
   containing Ipopt header files and libraries as ``IPOPT_INCLUDE_DIRS`` and
   ``IPOPT_LIB_DIRS`` in the ``setup.py`` file.

However, users (especially users on Windows) are strongly encouraged to use the
Anaconda installation instructions for pycalphad instead. ``conda install pycalphad``
and ``conda update pycalphad`` will automatically install Ipopt on all platforms.

After installation
~~~~~~~~~~~~~~~~~~

``TypeError: argument is not an mpz`` during a calculation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This is an upstream bug in sympy, where floats are unable to be pickled.
The fix has been completed, but not yet released. While the fix is not released,
removing the gmpy2 package from their Python environment (e.g.
``conda remove --force gmpy2``) will fix the error. Alternatively, setting the
environment variable ``MPMATH_NOGMPY`` to a non-zero value will fix the error.

.. _Anaconda: https://anaconda.com/download
.. _`Jupyter Notebook`: http://jupyter.readthedocs.io/en/latest/index.html
.. _Ipopt: https://projects.coin-or.org/Ipopt
.. _SymEngine: https://github.com/symengine/symengine
