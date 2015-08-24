Installation Instructions
=========================

Windows (Anaconda)
------------------
The Anaconda_ scientific Python distribution by Continuum Analytics is recommended
for Windows users. After you have installed either Anaconda or Miniconda, use
``conda install -c https://conda.anaconda.org/richardotis pycalphad`` to install.

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

Prerelease Versions (Advanced Users)
------------------------------------
* ``git clone https://github.com/richardotis/pycalphad.git``
* ``python setup.py develop`` inside a virtualenv

.. _Anaconda: http://continuum.io/downloads/
