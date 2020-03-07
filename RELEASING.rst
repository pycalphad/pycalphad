Releasing pycalphad
===================

When releasing a new version of pycalphad:

1. All pull requests / issues tagged with the upcoming version milestone should be resolved or deferred.
2. ``git pull`` to make sure you haven't missed any last-minute commits. **After this point, nothing else is making it into this version.**
   A minor release can be done later if something important is missed.
3. Ensure that all tests pass locally on develop. Feature tests which are deferred to a future
   milestone should be marked with the ``SkipTest`` decorator.
4. Regenerate the API documentation with ``sphinx-apidoc -o docs/api/ pycalphad/``
5. Resolve differences and commit the updated API documentation to the develop branch of the repository.
6. ``git push`` and verify all tests pass on all CI services.
7. Generate a list of commits since the last version with ``git log --oneline --decorate --color 0.1^..origin/develop``
   Replace ``0.1`` with the tag of the last public version.
8. Condense the change list into something user-readable. Update and commit CHANGES.rst with the release date.
9. ``git checkout master``

   ``git merge develop`` (merge commits unnecessary for now)
10. ``git stash``

   ``git tag 0.2 master -m "Version 0.2"`` Replace ``0.2`` with the new version.

   ``git show 0.2`` to ensure the correct commit was tagged.

   ``git push origin master --tags``

   ``git stash pop``
11. The new version is tagged in the repository. Now the public package must be built and distributed.

Uploading to PyPI
-----------------
1. ``rm -R dist/*`` on Linux/OSX or ``del dist/*`` on Windows
2. With the commit checked out which was tagged with the new version:
   ``python setup.py sdist``

   **Make sure that the script correctly detected the new version exactly and not a dirty / revised state of the repo.**

   Assuming a correctly configured .pypirc:

   ``twine upload -r pypi -u rotis dist/*``

Uploading to conda-forge
------------------------
Start with the commit checked out which was tagged with the new version.

1. Generate the SHA256 hash of the build artifact (tarball) submitted to PyPI.
2. Fork the conda-forge/pycalphad-feedstock repo.
3. Update pycalphad version and sha256 strings in the ``recipe/meta.yaml`` file.
4. If any of the dependencies changed since the last release, make sure to update the ``recipe/meta.yaml`` file.
5. Submit a pull request to the main pycalphad feedstock repo.
6. Once the tests pass, merge the pull request.
