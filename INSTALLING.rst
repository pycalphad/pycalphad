Installation Instructions
=========================


This page will guide you through installing pycalphad and the `Jupyter Notebook`_
application, which is suggested for running pycalphad interactively.

Anaconda (recommended)
----------------------

For all Windows, macOS and Linux platforms, it is recommended to use Anaconda_
to install the latest release of pycalphad. Anaconda is a scientific Python
distribution by Anaconda, Inc. It provides good support for various
scientific packages and otherwise challenging to install packages.

To install pycalphad from Anaconda

1. Download and install Anaconda_
2. From the Anaconda Prompt (Windows) or a terminal emulator (macOS and Linux) run ``conda config --add channels conda-forge; conda config --add channels pycalphad``
3. Run the ``conda install pycalphad`` command to install pycalphad

PyPI
----

Release versions of pycalphad are available on PyPI. As of pycalphad 0.6,
the `Ipopt`_ library is used for numerical optimization and must be installed
separately in order to install pycalphad from PyPI. Instructions for downloading
and installing Ipopt are found at https://www.coin-or.org/Ipopt/documentation/node10.html.

NumPy, SciPy and Cython are all *build* requirements of pycalphad and must be
installed before you install pycalphad.

To install pycalphad from PyPI using pip:

1. Download and install Ipopt
2. Run the ``pip install numpy scipy cython`` command in a terminal emulator
3. Run the ``pip install pycalphad`` command to install pycalphad
4. (Optional) run ``pip install jupyter`` to install the Jupyter Notebook application


Development Versions (Advanced Users)
-------------------------------------

To install a development version of pycalphad, you can use either an Anaconda or
vanilla Python distribution.

In either case, it is suggested to use a virtual environment. These instructions
will walk you through installing pycalphad in a virtual environment called
``pycalphad-dev``.

Anaconda
~~~~~~~~

From the Anaconda Prompt (Windows) or a terminal emulator (macOS or Linux)

1. Add the conda channels ``conda config --add channels conda-forge; conda config --add channels pycalphad``
2. Create the virtual environment and install pycalphad into it ``conda create -n pycalphad-dev pycalphad``
3. Remove the installed pycalphad package so the development version can be installed ``conda remove --force -n pycalphad-dev pycalphad``
4. Activate the environment ``activate pycalphad-dev`` (Windows) or ``source activate pycalphad-dev`` (macOS or Linux)
5. Get the pycalphad source ``git clone https://github.com/pycalphad/pycalphad.git pycalphad/`` (or download from https://github.com/pycalphad/pycalphad)
6. Go to the top level directory of the package ``cd pycalphad``
7. Run ``pip install -e .``

PyPI
~~~~

From the Anaconda Prompt (Windows) or a terminal emulator (macOS or Linux)

1. Download and install `Ipopt`_
2. Follow the instructions to install `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/install.html>`_
3. Make the virtual environment ``mkvirtualenv pycalphad-dev``
4. Activate the environment ``workon pycalphad``
5. Install the build requirements ``pip install numpy scipy cython``
6. Get the pycalphad source ``git clone https://github.com/pycalphad/pycalphad.git pycalphad/``
7. Go to the top level directory of the package ``cd pycalphad``
8. Run ``pip install -e .``

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
