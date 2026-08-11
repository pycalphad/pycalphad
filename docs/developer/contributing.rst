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
#. Tests are for verifying correctness.
#. Most tests should be small, fast, and test only one thing so if/when the test fails, it is obvious where the problem lies.

   a. Tests for ``Model`` (and subclass) implementations should prefer to use :py:mod:`pycalphad.tests.test_energy.check_output` or :py:mod:`pycalphad.tests.test_energy.check_energy` rather than using ``Workspace`` (or ``equilibrium``) as that also exercises the minimizer.
   b. Tests that exercise the minimizer (synonym: solver) should be marked ``@pytest.mark.solver``

#. PyCalphad provides custom `pytest fixtures <https://docs.pytest.org/en/stable/explanation/fixtures.html>`_ :py:mod:`pycalphad.tests.fixtures.select_database` and :py:mod:`pycalphad.tests.fixtures.load_database` for loading databases from ``pycalphad.tests.databases``. Prefer not to add new databases to the test suite if a feature or bug can be exercised with a database already present in the test suite.
