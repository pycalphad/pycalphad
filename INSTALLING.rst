pip (recommended)
=================

To install pycalphad from `PyPI <https://pypi.org/project/pycalphad/>`_ using pip:

.. code-block:: bash

   pip install -U pip setuptools
   pip install -U pycalphad

A recommended best practice is to install pycalphad into a virtual environment.
To create an environment called ``pycalphad-env`` on Linux and macOS/OSX:

.. code-block:: bash

   python -m venv pycalphad-env
   source pycalphad-env/bin/activate
   pip install -U pip setuptools
   pip install -U pycalphad

This environment must be activated every time you use pycalphad.

conda
=====

`Anaconda`_ is a distribution platform for scientific Python packages created by Anaconda, Inc.
It comes bundled with the ``conda`` package manager to install Python and non-Python packages.
If you don't have Anaconda already, you can download the `Miniconda`_ distribution.

To install pycalphad from `conda-forge <https://github.com/conda-forge/pycalphad-feedstock/>`_ using conda:

.. code-block:: bash

   conda install -c conda-forge pycalphad


Development version
===================

Installing
----------

The source code for the latest development version of pycalphad is available on `GitHub <https://github.com/pycalphad/pycalphad>`_.
You will need a working version of  `git`_ to download the source code.
Installing development versions of pycalphad also requires a working C++ compiler.

* **Windows:** Install `git`_ and `Microsoft Visual C++ Build Tools version 14.X <https://visualstudio.microsoft.com/downloads/>`_.
  If you are unfamiliar with the process, you can find a `tutorial here <https://beenje.github.io/blog/posts/how-to-setup-a-windows-vm-to-build-conda-packages/#developer-tools-installation>`_.

To install an editable version the latest development branch of pycalphad, run:

.. code-block:: bash

   git clone https://github.com/pycalphad/pycalphad.git
   cd pycalphad
   pip install -U pip setuptools
   pip install -U -r requirements-dev.txt
   pip install -U --no-build-isolation --editable .

Then run the automated tests to ensure everything installed correctly:

.. code-block:: bash

   pytest pycalphad

Upgrading
---------

Changes to Python files (``.py`` extension) in an editable install will be reflected as soon a file is saved.
Changes to `Cython`_ files (``.pyx`` and ``.pxd`` extensions) must be recompiled before they take effect.
The Cython files can be recompiled by running (from the root directory [#f1]_ of the project):

.. code-block:: bash

   python setup.py build_ext --inplace

By default, the development version installed will track the latest changes in
the ``develop`` branch of pycalphad
`on GitHub <https://github.com/pycalphad/pycalphad>`_.

To update the code in the current branch to the latest changes on GitHub, the
changes are pulled from GitHub by running ``git pull``.

To switch to a different branch, e.g. ``master`` (which tracks the latest
released version) or another feature branch, run ``git checkout <branch>``,
where ``<branch>`` is the name of the branch to checkout, such as ``master``
(without the brackets ``<`` and ``>``).


.. _Anaconda: https://anaconda.com/download
.. _Cython: https://cython.org/
.. _git: https://git-scm.com/
.. _`Jupyter Notebook`: http://jupyter.readthedocs.io/en/latest/index.html
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html

.. [#f1] The "root directory" is the the top level project directory containing the ``pyproject.toml`` file, the ``pycalphad/`` package directory, etc.