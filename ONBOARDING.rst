Onboarding as a Developer
=========================

Workplace setup & software tutorials
------------------------------------

Unless you already have them, install VSCode (or other IDE for Python), Git, Build Tools for Visual Studio version 14.X, and uv.

You can follow a tutorial for all these but uv at `developer tools installation <https://beenje.github.io/blog/posts/how-to-setup-a-windows-vm-to-build-conda-packages/#developer-tools-installation>`_. Follow the article up until "Testing", which is not essential but useful if you want to check that you installed everything properly.

For uv, you can follow the installation instructions on Astral website: `uv installation <https://docs.astral.sh/uv/getting-started/installation/>`_.

Make sure to download for the respective operating systems of the computer.

Make sure `Microsoft Visual C++ Build Tools <https://visualstudio.microsoft.com/downloads/?q=build+tools#build-tools-for-visual-studio-2026>`_ is installed because it is essential towards installing pycalphad properly.

After completing those downloads, `click <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`_ to download Microsoft’s Python extension.

As well, make sure your computer has `Python website`_ 3.11+ (or a more updated version) downloaded.
Now, you can install the latest development branch of pycalphad. Run this command in Git Bash:

.. code-block:: bash

   git clone https://github.com/pycalphad/pycalphad.git

Next, you want to create a virtual environment using uv (`<https://docs.astral.sh/uv/pip/environments/>`_).

How to create your virtual environment using uv
-----------------------------------------------

Create your virtual environment with uv and Python 3.13 (or a more updated version) using the command below:

.. code-block:: bash

    uv venv --python 3.13

Install dependencies and pycalphad
----------------------------------

.. code-block:: bash

   uv sync

Run tests
---------

.. code-block:: bash

   uv run pytest pycalphad

After the pytests are done running, your terminal should be similar to this:

.. code-block:: bash

    ========== 309 passed, 2 skipped, 1 xfailed in 70.21s (0:01:10) ==========

Setting your VSCode environment
-------------------------------

We want to select the correct interpreter/environment in Visual Studio Code. Click into any file in the pycalphad codebase, and in the bottom right corner, you will be able to click on the Python version.

At the top of Visual Studio, you will see a drop-down menu- click there and select the interpreter that has the name of your environment. In this case, it would be named something like this: Python 3.13 (‘pycalphad’).

Setting up your fork and remote repositories
--------------------------------------------

Now that you've written your code contributions, you are ready to push the code changes to GitHub. Here's how to set up your remotes from the main code base.

#. Run ``uv run pytest pycalphad`` in your terminal to check your code changes locally.
#. Read `here <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_ to figure out how to create a fork from the original pycalphad repository in GitHub.
#. Read in `GitHub documents`_ to configure an additional remote upstream repository that syncs with the fork.

Create another remote repository that points upstream to your fork.

Our end result is that we want our origin repository to fetch and push to your fork while our upstream repository will fetch and push to the original pycalphad repository. The command ``git remote -v`` lists the current configured remote repositories for your fork.

Like so:

.. code-block:: bash

    user@MT-102934 MINGW64 ~/pycalphad (branch_name)
    $ git remote -v
    origin     https://github.com/username/forkname.git (fetch)
    origin     https://github.com/username/forkname.git (push)
    upstream   https://github.com/pycalphad/pycalphad.git (fetch)
    upstream   https://github.com/pycalphad/pycalphad.git (push)

If either your origin or upstream does not match the above links, use the commands below:

To remove a remote: ``git remote rm <remote-name>``

Add remote: ``git remote add <upstream or origin> <remote-name>``

You can get your forks HTTPS from the GitHub website. Clicking the down arrow on the forks button should show you your fork. Go to your fork and click on the green "Code" button which should give you the HTTPS.

Congratulations! You are now ready to contribute to pycalphad!

Building a local version of the Pycalphad website
-------------------------------------------------

To build a local copy of the pycalphad website using the code repository, follow the instructions below. This is helpful for when you write a pull request because you want to update the documentation at the same time that you add a new feature. This way you can test changes to the documentation without having to push them to GitHub.

1. Install pandoc by following the instructions on the `pandoc website`_.

2. Run the command below in your terminal to build the documentation locally.

.. code-block:: bash

    bash tools/build-docs.sh

3. Or use command `Run tasks in VS Code`_ and select the task named ``build and serve docs`` to run the same command as above.

4. After a few seconds, you should receive a message saying ``The HTML pages are in docs/_build/html``.

5. Navigate to the URL in the parentheses after ``Serving HTTP on ::``  in the terminal and that should be your local copy of the pycalphad website. The given URL is https://[::]:8000/ in the parentheses. Replace the [::] with localhost to give the actual URL: http://localhost:8000/ which will result in your own local version of the pycalphad website.

How to recompile Pycalphad
--------------------------

Run this command in your terminal. Make sure any Jupyter kernels that are using pycalphad are shut down, other you will see "access denied" errors.

.. code-block:: bash

    python setup.py develop --no-deps

Troubleshooting
---------------

If you get an error in any of the above steps, try the following:

- Make sure you have the latest version of Python installed. You can check your Python version by running ``python --version`` in your terminal. If you have an older version, you can download the latest version from the `Python website`_.

- Make sure you have the latest version of Git installed. You can check your Git version by running ``git --version`` in your terminal. If you have an older version, you can download the latest version from the `Git website`_.

- Make sure you have the latest version of uv installed. You can check your uv version by running ``uv --version`` in your terminal. If you have an older version, you can download the latest version from the `uv website`_.

- Make sure you have the latest version of Microsoft Visual C++ Build Tools installed. You can check if you have it installed by going to "Add or Remove Programs" on your computer and looking for "Microsoft Visual C++ Build Tools". If you don't have it installed, you can download it from the `Microsoft website`_.

- Make sure you have your virtual environment and dependencies properly set up.

- If you are still having issues, try searching for the error message online or asking for help on the `pycalphad GitHub repository <https://github.com/pycalphad/pycalphad/issues>`_ or in the `pycalphad Gitter channel <https://gitter.im/pycalphad/pycalphad>`_.

.. _Python website: https://www.python.org/downloads/
.. _Git website: https://git-scm.com/install/
.. _uv website: https://docs.astral.sh/uv/getting-started/installation/
.. _Microsoft website: https://visualstudio.microsoft.com/downloads/?q=build+tools#build-tools-for-visual-studio-2026
.. _pandoc website: https://pandoc.org/installing.html
.. _GitHub documents: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-for-a-fork
.. _Run tasks in VS Code: https://code.visualstudio.com/docs/debugtest/tasks