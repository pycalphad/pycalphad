Installation Instructions
=========================

Windows (Anaconda)
------------------
The Anaconda_ scientific Python distribution by Continuum Analytics is recommended
for Windows users. After you have installed either Anaconda or Miniconda, use
``conda install -c richardotis pycalphad`` to install.
To install the package into an isolated environment, use ``conda create -c richardotis -n _envname_ pycalphad``
Then use ``source activate _envname_`` on Linux/OSX or ``activate _envname_`` on Windows to enter the environment.

Mac OSX
-------
If not using a special distribution like Canopy or Anaconda_, it's recommended to install
pycalphad in a virtualenv using ``virtualenvwrapper``.
``pip install pycalphad`` inside the virtualenv will install with any required dependencies.
You may also want to ``pip install fastcache`` for a mild performance boost.
If you are using Anaconda, see the Windows instructions.

Linux
-----
If not using a special distribution like Canopy or Anaconda_, it's recommended to install
pycalphad in a virtualenv using ``virtualenvwrapper``.
``pip install pycalphad`` inside the virtualenv will install with any required dependencies.
You may also want to ``pip install fastcache`` for a mild performance boost.
If you are using Anaconda, see the Windows instructions.

Development Versions (Advanced Users)
-------------------------------------
* ``git clone https://github.com/richardotis/pycalphad.git pycalphad/``
* Using conda:
    * ``conda create -c richardotis -n _envname_ pycalphad``
    * ``conda develop -n _envname_ pycalphad/``
    * ``source activate _envname_`` on Linux/OSX or ``activate _envname_`` on Windows to enter the environment.
* Or, inside a virtualenv: ``python setup.py develop``

.. _Anaconda: http://continuum.io/downloads/
