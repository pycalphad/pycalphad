Releasing pycalphad
===================

When releasing a new version of pycalphad:

1. All pull requests / issues tagged with the upcoming version milestone should be resolved or deferred.
2. Remove all .rst files from the docs/api directory.
3. Regenerate the API documentation with ``sphinx-apidoc -o docs/api/ pycalphad/``
4. Commit the updated API documentation to the master branch of the repository.
5. Generate a list of commits since the last version with ``git log --oneline --decorate --color 0.1^..origin/master``
   Replace ``0.1`` with the tag of the last public version.
6. Condense the change list into something user-readable. Update and commit CHANGES.rst with the release date.
