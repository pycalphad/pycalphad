Installation Instructions
=========================

Windows (Anaconda)
------------------
The Anaconda_ scientific Python distribution by Continuum Analytics is recommended
for Windows users. You can use pycalphad with Python 2 or Python 3, but we recommend
Python 3 for the best experience. After you have installed either Anaconda or Miniconda, use
``conda config --add channels richardotis conda-forge`` followed by
``conda install pycalphad`` to install. Note that you will need to have a working
C/C++ compiler for pycalphad to work, so ``conda install mingw`` may also be necessary on Windows.
To install the package into an isolated environment, use ``conda create -c richardotis -n [envname] pycalphad``
Then use ``source activate [envname]`` on Linux/OSX or ``activate [envname]`` on Windows to enter the environment.

For interactive pycalphad sessions, we recommend installing the `Jupyter Notebook`_.

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
* ``git clone https://github.com/pycalphad/pycalphad.git pycalphad/``
* Using conda:
    * ``conda config --add channels richardotis conda-forge``
    * ``conda create -n [envname] pycalphad``
    * ``conda develop -n [envname] pycalphad/``
    * ``source activate [envname]`` on Linux/OSX or ``activate [envname]`` on Windows to enter the environment.
* Or, inside a virtualenv: ``python setup.py develop``

.. _Anaconda: http://continuum.io/downloads/
.. _`Jupyter Notebook`: http://jupyter.readthedocs.org/en/latest/install.html
