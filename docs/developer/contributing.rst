Contributing to PyCalphad
=========================

This is a guide for best practices on making contributions to PyCalphad.
See :doc:`onboarding` for instructions on setting up your development environment.

Tests
-----

.. image:: https://codecov.io/gh/pycalphad/pycalphad/branch/develop/graph/badge.svg?token=Fu7FJZeJu0
    :target: https://codecov.io/gh/pycalphad/pycalphad
    :alt: Test Coverage

The PyCalphad test suite is one of the most valuable cumulative contributions of the library that ensures quality and continuous improvement.

1. PyCalphad tests aim for high code coverage. Ideally every PR has a net improvement on test coverage without trivially exercising code paths to improve coverage metrics.
#. Tests are for verifying correctness. Prefer testing with real databases.
#. Most tests should be small, fast, and test only one thing so if/when the test fails, it is obvious where the problem lies.

   a. Tests for ``Model`` (and subclass) implementations should prefer to use :py:mod:`pycalphad.tests.test_energy.check_output` or :py:mod:`pycalphad.tests.test_energy.check_energy` rather than using ``Workspace`` (or ``equilibrium``) as that also exercises the minimizer.
   b. Tests that exercise the minimizer (synonym: solver) should be marked ``@pytest.mark.solver``

#. PyCalphad provides custom `pytest fixtures <https://docs.pytest.org/en/stable/explanation/fixtures.html>`_ :py:mod:`pycalphad.tests.fixtures.select_database` and :py:mod:`pycalphad.tests.fixtures.load_database` for loading databases from ``pycalphad.tests.databases``. Prefer not to add new databases to the test suite if a feature or bug can be exercised with a database already present in the test suite.

Contribution Workflow
---------------------

All changes are made through `GitHub pull requests <https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request>`_ and are reviewed by a maintainer before merging.
Before submitting a pull request, please `open an issue <https://github.com/pycalphad/pycalphad/issues>`_ or comment on an existing relevant one to discuss the changes you plan to make.
A little alignment upfront prevents significant re-work later.

AI Policy
---------

**Contributors may use any tools they would like to craft contributions, but there must be a human-in-the-loop.**

The PyCalphad project values human contributors more than their contributions.
Maintainers/Reviewers expect to engage with humans, not with AI.
We encourage new contributors to _not_ use AI as they learn the project.

Human contibutors, are responsible for:

- authoring issue and pull request descriptions
- understanding every change
- reviewing and testing the changes, and
- ensuring the changes are correct (to the best of your ability)

Please read `Numba's AI Tool Use Policy <https://numba.readthedocs.io/en/latest/reference/ai_tools_policy.html>`_, which informed our policy, for additional details and rationale.