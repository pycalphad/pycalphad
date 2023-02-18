Onboarding as a Developer
=========================

Workplace setup & software tutorials
------------------------------------

Unless you already have them, install VSCode (or other IDE for Python), Git, Build Tools for Visual Studio version 14.X, and Miniconda. You can follow a tutorial for all these `downloads <https://beenje.github.io/blog/posts/how-to-setup-a-windows-vm-to-build-conda-packages/#developer-tools-installation>`_. Follow the article up until "Testing", which is not essential but useful if you want to check that you installed everything properly. Make sure to download for the respective operating systems of the computer. Make sure Microsoft Visual C++ Build Tools is installed because it is essential towards installing pycalphad properly.
After completing those downloads, `click <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`_ to download Microsoft’s Python extension. 
As well, make sure your computer has `Python`_ 3.7+ (or a more updated version) downloaded. 
Now, you can install the latest development branch of pycalphad. Run this command in Git Bash:

.. code-block:: bash

   git clone https://github.com/pycalphad/pycalphad.git

Next, you want to create a conda environment. 

How to create your conda environment
------------------------------------

Create your conda virtual environment using the application called Anaconda Prompt. You can find this by searching your computer in the bottom left bar.
 After ``-n`` is where we specify our environment name (in this case the name is ‘calphad’), and then the version of Python we want to install.

``conda create -n calphad python=3.9``

Activate your conda environment in Anaconda Prompt
--------------------------------------------------

After running ``conda activate <virtual environment name>``, you should see the ``(base)`` changed to ``(your environment name)``, which means we are now executing from within that active conda environment.

.. code-block:: bash

    (base) C:>conda activate calphad
    (calphad) C:>

Now run these commands in Anaconda Prompt:

.. code-block:: bash

   conda install pip
   cd pycalphad
   pip install -U pip setuptools
   pip install -U -r requirements-dev.txt
   pip install -U --no-build-isolation --editable .

Then ``cd ..`` out of the pycalphad path. Now, run the built-in tests using ``pytest pycalphad`` to ensure pycalphad is correctly installed.

After the pytests are done running, your Anaconda Prompt display should be similar to this:

.. code-block:: bash

    ============ 208 passed, 2 skipped, 1 xfailed, 105 warnings in 70.34s (0:02:43) ============

Setting your VSCode environment
-------------------------------

We want to select the correct interpreter/environment in Visual Studio Code. Click into any file in the pycalphad codebase, and in the bottom right corner, you will be able to click on the Python version.

At the top of Visual Studio, you will see a drop-down menu- click there and select the interpreter that has the name of your environment. In this case, it would be named something like this: Python 3.913 (‘calphad’).

Setting up your fork and remote repositories
--------------------------------------------

Now that you've written your code contributions, you are ready to push the code changes to GitHub. Here's how to set up your remotes from the main code base.

#. Run ``pytest pycalphad`` in Anaconda Prompt (in your active conda environment) to check your code changes locally.
#. Read `here <https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_ to figure out how to create a fork from the original pycalphad repository in GitHub.
#. Read in `GitHub documents`_ to configure an additional remote upstream repository that syncs with the fork.

Create another remote repository that points upstream to your fork.

Our end result is that we want our origin repository to fetch and push to your fork while our upstream repository will fetch and push to the original pycalphad respoitory. The command ``git remote -v`` lists the current configured remote repositories for your fork.

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

Below are numbered action items to build a local copy of the pycalphad website using the code repository. This is helpful for when you write a pull request because you want to update the documentation at the same time that you add a new feature. This way you can test changes to the documentation without having to push them to GitHub.

1. Activate your conda environment (review steps above) in Anaconda Prompt, then install Pandoc through your named conda environment using

.. code-block:: bash

    conda install -n calphad -c conda-forge Pandoc

2. After installing Pandoc, ``cd <forkname>`` to go into your pycalphad fork directory. To check if your fork exists, use ``dir``
#. From the main directory of your fork, run

.. code-block:: bash

    sphinx-build -W -b html docs docs/_build/html

4. After a few seconds, you should receive a message saying ``The HTML pages are in docs/_build/html``. Now have Python serve the directory as a website using

.. code-block:: bash

    python -m http.server --directory docs/_build/html

5. Navigate to the URL in the parentheses after ``Serving HTTP on ::``  in the terminal and that should be your local copy of the pycalphad website. The given URL is https://[::]:8000/ in the parentheses. Replace the [::] with localhost to give the actual URL: http://localhost:8000/ which will result in your own local version of the pycalphad website.

How to recompile Pycalphad
--------------------------

Run this command in your activated conda environment. Make sure any Jupyter kernels that are using pycalphad are shut down, other you will see "access denied" errors.

.. code-block:: bash

    python setup.py develop --no-deps

Potential errors and their solutions
------------------------------------

If Anaconda Prompt displays an error similar to

.. code-block:: bash

    Application error:
    Cannot find source directory

after trying to run ``sphinx-build``, check to make sure that your fork is compiled by running

.. code-block:: bash

    pip install -U --no-build-isolation --editable .

.. _Python: https://www.python.org/downloads/
.. _GitHub documents: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-for-a-fork