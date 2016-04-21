Releasing pycalphad
===================

When releasing a new version of pycalphad:

1. All pull requests / issues tagged with the upcoming version milestone should be resolved or deferred.
2. Ensure that all tests pass locally on develop. Feature tests which are deferred to a future
   milestone should be marked with the ``SkipTest`` decorator.
3. Regenerate the API documentation with ``sphinx-apidoc -o docs/api/ pycalphad/``
4. Resolve differences and commit the updated API documentation to the develop branch of the repository.
5. ``git push`` and verify all tests pass on all CI services.
6. Generate a list of commits since the last version with ``git log --oneline --decorate --color 0.1^..origin/develop``
   Replace ``0.1`` with the tag of the last public version.
7. Condense the change list into something user-readable. Update and commit CHANGES.rst with the release date.
8. If you have Sphinx installed in a virtual environment with pycalphad:
   Run ``sphinx-build -b html ~/git/pycalphad/docs docs/_build/html`` to do a spot check on the docs before pushing.
9. ``git checkout master``

   ``git merge develop`` (merge commits unnecessary for now)
10. ``git pull`` to make sure you haven't missed any last-minute commits. **After this point, nothing else is making it into this version.**
   A minor release can be done later if something important is missed.
11. ``git stash``

   ``git tag -s 0.2 master -m "Version 0.2"`` Replace ``0.2`` with the new version. pycalphad should be signed with GPG key **98628A70**.
   If you are using a hardware token on Linux, you may need to ``killall -1 gpg-agent`` for it to be detected.

   ``git show 0.2`` to ensure the correct commit was tagged and signed

   ``git tag -v 0.2`` to verify the GPG signature

   ``git push origin master --tags``

   ``git stash pop``
12.The new version is tagged in the repository. Now the public package must be built and distributed.

Uploading to PyPI
-----------------
1. ``rm -R dist/*`` on Linux/OSX or ``del dist/*`` on Windows
2. With the commit checked out which was tagged with the new version:
   ``python setup.py sdist bdist_wheel``

   **Make sure that the script correctly detected the new version exactly and not a dirty / revised state of the repo.**

   Assuming a correctly configured .pypirc:
   ``twine upload -r pypi --sign -u rotis -i 0161A98D dist/*``

Uploading to Anaconda.org
-------------------------
Start with the commit checked out which was tagged with the new version.

1. ``rm /home/rotis/anaconda/conda-bld/linux-64/pycalphad-*.tar.bz2`` on Linux/OSX (use ``del`` and correct path on Windows)
2. ``rm -R dist/*`` on Linux/OSX or ``del dist/*`` on Windows
3. ``conda build --python 2.7 conda_recipe/``

   ``conda build --python 3.3 conda_recipe/``

   ``conda build --python 3.4 conda_recipe/``

   ``conda build --python 3.5 conda_recipe/``

4. ``conda convert --platform all /home/rotis/anaconda/conda-bld/linux-64/pycalphad-*.tar.bz2 -o ./dist``
5. ``anaconda upload -u richardotis dist/*/*``