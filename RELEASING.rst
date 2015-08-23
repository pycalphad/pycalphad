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
9. ``git tag -s 0.2 master -m "Version 0.2"`` Replace ``0.2`` with the new version. pycalphad should be signed with GPG key **0161A98D**.

   ``git show 0.2`` to ensure the correct commit was tagged and signed

   ``git tag -v 0.2`` to verify the GPG signature

   ``git push origin master --tags``
10.The new version is tagged in the repository. Now the public package must be built and distributed.