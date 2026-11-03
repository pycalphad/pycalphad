The recommended method of installing pycalphad depends on your preferred workflow.
The common workflows can roughly be broken down into the following categories:

* **Project-based** (e.g. ``uv``, ``pixi``) *(recommended for new users)*
* **Environment-based** (e.g. ``pip``, ``conda``) *(the traditional workflow)*
* **Development version** (building from source) *(for contributing to pycalphad)*

.. tab-set::

   .. tab-item:: Project-based

      .. tab-set::

         .. tab-item:: uv

            Here is a step-by-step guide to setting up a project to use pycalphad with `uv`_, a Python package manager.

            #. Install uv by following `the instructions in the uv documentation <https://docs.astral.sh/uv/getting-started/installation/>`__.

            #. Create a new project in a new subdirectory by running the following in a terminal:

               .. code-block:: bash

                  uv init try-pycalphad
                  cd try-pycalphad

               The second command changes directory into the directory of your project.

            #. Add pycalphad to your project:

               .. code-block:: bash

                  uv add pycalphad

               This will automatically install Python if you don't already have it installed.

            #. Try out pycalphad!

               .. code-block:: bash

                  uv run python

               This will launch a Python interpreter session, from which you can ``import pycalphad``.

            .. tip::

               You can install other Python libraries in the same way.
               For example, to run the pycalphad examples in Jupyter notebooks:

               .. code-block:: bash

                  uv add jupyterlab
                  uv run jupyter lab

            .. note::

               After rebooting your computer, you'll want to navigate to your ``try-pycalphad`` project directory and run ``uv run python`` to drop back into a Python interpreter with pycalphad importable.
               To run a Python script, you can use ``uv run myscript.py``.

               Read more in `the uv guide to working on projects <https://docs.astral.sh/uv/guides/projects/>`__.

         .. tab-item:: pixi

            If you work with non-Python packages, you may prefer to install pycalphad as a `conda`_ package, so that you can use the same workflow for packages which are not available on `PyPI <https://pypi.org/>`__, the Python Package Index.
            Conda can manage packages in any language, so you can use it to install Python itself, compilers, and other languages.

            The steps to install pycalphad from `conda-forge`_ using `pixi`_ are very similar to the steps for uv:

            #. Install pixi by following `the instructions in the pixi documentation <https://pixi.sh/latest/installation/>`__.

            #. Create a new project in a new subdirectory:

               .. code-block:: bash

                  pixi init try-pycalphad
                  cd try-pycalphad

            #. Add pycalphad to your project:

               .. code-block:: bash

                  pixi add pycalphad

            #. Try out pycalphad!

               .. code-block:: bash

                  pixi run python

            .. tip::

               You can install other packages in the same way.
               For example, to run the pycalphad examples in Jupyter notebooks:

               .. code-block:: bash

                  pixi add jupyterlab
                  pixi run jupyter lab

   .. tab-item:: Environment-based

      In project-based workflows, a project is a directory containing a manifest file describing the project, a lock file describing the exact dependencies of the project, and the project's (potentially multiple) environments.
      In contrast, in environment-based workflows you install packages into an environment, which you can activate and deactivate from any directory.
      These workflows are well-established, but lack some reproducibility benefits of project-based workflows.

      .. tab-set::

         .. tab-item:: pip

            #. `Install Python <https://www.python.org/downloads/>`__.

            #. A recommended best practice is to install pycalphad into a virtual environment.
               To create and activate an environment called ``pycalphad-env`` on Linux and macOS:

               .. code-block:: bash

                  python -m venv pycalphad-env
                  source pycalphad-env/bin/activate

               On Windows, activate the environment by running ``pycalphad-env\Scripts\activate`` instead.
               This environment must be activated every time you use pycalphad.

            #. Install pycalphad from `PyPI`_ using `pip`_:

               .. code-block:: bash

                  python -m pip install -U pycalphad

         .. tab-item:: conda

            `conda`_ is a package manager that can install Python and non-Python packages.
            If you don't have conda already, `Miniforge`_ is the recommended way to install it.

            To create an environment called ``pycalphad-env`` with pycalphad installed from `conda-forge`_:

            .. code-block:: bash

               conda create -n pycalphad-env -c conda-forge pycalphad
               conda activate pycalphad-env

            This environment must be activated every time you use pycalphad.

   .. tab-item:: Development version

      The source code for the latest development version of pycalphad is available on `GitHub <https://github.com/pycalphad/pycalphad>`__.
      Development versions are built from source and installed in editable mode, which is mainly needed by users who wish to contribute to pycalphad.

      .. tab-set::

         .. tab-item:: Quickstart

            You will need a working version of `git`_ to download the source code and `uv`_ to install pycalphad and its dependencies.
            Installing development versions of pycalphad also requires a working C++ compiler.

            * **Windows:** Install `git`_ and `Microsoft Visual C++ Build Tools version 14.X <https://visualstudio.microsoft.com/downloads/>`__.
              If you are unfamiliar with the process, you can find a `tutorial here <https://beenje.github.io/blog/posts/how-to-setup-a-windows-vm-to-build-conda-packages/#developer-tools-installation>`__.
            * **macOS:** Install the Xcode Command Line Tools by running ``xcode-select --install``.

            To install an editable version of the latest development branch of pycalphad, run:

            .. code-block:: bash

               git clone https://github.com/pycalphad/pycalphad.git
               cd pycalphad
               uv sync

            Then run the automated tests to ensure everything installed correctly:

            .. code-block:: bash

               uv run pytest

         .. tab-item:: Detailed instructions

            The :doc:`onboarding guide <developer/onboarding>` has detailed instructions for setting up a development environment, including installing the prerequisite tools, configuring VSCode, setting up your fork and remote repositories, building a local version of this website, and troubleshooting.
            Please read the :doc:`contributing guide <developer/contributing>` before submitting a pull request.

            .. rubric:: Upgrading

            Changes to Python files (``.py`` extension) in an editable install will be reflected as soon as a file is saved.
            Changes to `Cython`_ files (``.pyx`` and ``.pxd`` extensions) must be recompiled before they take effect.
            uv recompiles the Cython files automatically the next time you run ``uv sync`` or ``uv run``.

            By default, the development version installed will track the latest changes in the ``develop`` branch of pycalphad `on GitHub <https://github.com/pycalphad/pycalphad>`__.

            To update the code in the current branch to the latest changes on GitHub, the changes are pulled from GitHub by running ``git pull``.

            To switch to a different branch, e.g. ``master`` (which tracks the latest released version) or another feature branch, run ``git checkout <branch>``, where ``<branch>`` is the name of the branch to checkout, such as ``master`` (without the brackets ``<`` and ``>``).

See next steps in the :doc:`examples <examples/1_Phase_Diagrams/index>`.
Questions about installing and using pycalphad can be addressed on the `PyCalphad Discussion board <https://github.com/pycalphad/pycalphad/discussions>`__.


.. _conda: https://docs.conda.io/projects/conda/en/latest/index.html
.. _conda-forge: https://conda-forge.org/
.. _Cython: https://cython.org/
.. _git: https://git-scm.com/
.. _Miniforge: https://conda-forge.org/download/
.. _pip: https://pip.pypa.io/en/stable/getting-started/
.. _pixi: https://pixi.sh/latest/
.. _PyPI: https://pypi.org/project/pycalphad/
.. _uv: https://docs.astral.sh/uv/
