Releasing pycalphad
===================

When releasing a new version of pycalphad:

1. All pull requests / issues tagged with the upcoming version milestone should be resolved or deferred.
2. Remove all .rst files from the docs/api directory.
3. Regenerate the API documentation with ``sphinx-apidoc -o docs/api/ pycalphad/``.
4. Commit the updated API documentation to the repository.