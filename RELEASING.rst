Releasing pycalphad
===================

When releasing a new version of pycalphad:

1. All pull requests / issues tagged with the upcoming version milestone should be resolved or deferred.
2. Regenerate the API documentation with ``sphinx-apidoc -o docs/api/ pycalphad/``
3. Resolve differences and commit the updated API documentation to the master branch of the repository.
4. Generate a list of commits since the last version with ``git log --oneline --decorate --color 0.1^..origin/master``
   Replace ``0.1`` with the tag of the last public version.
5. Condense the change list into something user-readable. Update and commit CHANGES.rst with the release date.
6. If you have Sphinx installed in a virtualenv with pycalphad, change to the docs directory.
   Run ``sphinx-build -b html . _build/html`` to do a spot check on the docs before pushing.
