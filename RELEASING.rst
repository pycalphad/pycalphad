Releasing pycalphad
===================

When releasing a new version of pycalphad:

1. All pull requests / issues tagged with the upcoming version milestone should be resolved or deferred.
2. Check Travis-CI and ensure that all tests pass on master. Feature tests which are deferred to a future
   milestone should be marked with the ``SkipTest`` decorator.
3. Regenerate the API documentation with ``sphinx-apidoc -o docs/api/ pycalphad/``
4. Resolve differences and commit the updated API documentation to the master branch of the repository.
5. Generate a list of commits since the last version with ``git log --oneline --decorate --color 0.1^..origin/master``
   Replace ``0.1`` with the tag of the last public version.
6. Condense the change list into something user-readable. Update and commit CHANGES.rst with the release date.
7. If you have Sphinx installed in a virtualenv with pycalphad, change to the docs directory.
   Run ``sphinx-build -b html . _build/html`` to do a spot check on the docs before pushing.
8. ``git pull`` to make sure you haven't missed any last-minute commits. **After this point, nothing else is making it into this version.**
   A minor release can be done later if something important is missed.
9. ``git stash``

   ``git tag -s 0.2 master -m "Version 0.2"`` Replace ``0.2`` with the new version. pycalphad should be signed with GPG key **0161A98D**.
   If you are using a hardware token on Linux, you may need to ``killall -1 gpg-agent`` for it to be detected.

   ``git show 0.2`` to ensure the correct commit was tagged and signed

   ``git tag -v 0.2`` to verify the GPG signature

   ``git push origin master --tags``

   ``git stash pop``
10.The new version is tagged in the repository. Now the public package must be built and distributed.

Uploading to PyPI
-----------------
1. Delete all old files from the dist directory, if necessary.
2. With the commit checked out which was tagged with the new version:
   ``python setup.py sdist bdist_wheel``

   **Make sure that the script correctly detected the new version exactly and not a dirty / revised state of the repo.**

   Assuming a correctly configured .pypirc:
   ``twine upload -r pypi --sign -u rotis -i 0161A98D dist/*``

Uploading to Anaconda.org
-------------------------
Eventually we'd like to pull directly from GitHub using tags. This is a temporary solution.
These instructions are adapted from https://github.com/menpo/menpo/wiki/Build-pure-Python-conda-package-from-PyPI-for-all-platforms,-2.7-3.4

1. In a directory not in the repository, after pushing to PyPI: ``conda skeleton pypi pycalphad``
2. Modify the meta.yaml file to have ``nose`` and ``mock`` in the build dependencies.
3. ``conda build --python 2.7 ./pycalphad``
4. ``conda build --python 3.3 ./pycalphad``
5. ``conda build --python 3.4 ./pycalphad``
6. ``conda convert --platform all /home/rotis/anaconda/conda-bld/linux-64/pycalphad-0.2-py*.tar.bz2 -o ./out``
   Replace 0.2 with the new version.
7. ``anaconda upload -u richardotis ./out/*/pycalphad-0.2-*.tar.bz2``
