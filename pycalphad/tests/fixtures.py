from importlib.resources import files
import pytest
import numpy as np
import pycalphad.tests.databases
from pycalphad.io.database import Database
from pycalphad.core.solver import Solver, SolverResult

@pytest.fixture(scope="session")
def load_database(request):
    """
    Helper fixture to load a database (parameterized by the value of `request`).
    """
    db = Database(files(pycalphad.tests.databases).joinpath(request.param))
    def _load_database():
        return db
    return _load_database


def select_database(path):
    """
    Decorator to facilitate safe, fast loading of database objects. Use as

    ```
    @select_database(\"filename.tdb\")  # matches a file in the pycalphad.test.databases directory
    def test_name_of_my_test(load_database):
        dbf = load_database()  # equivalent to `dbf = Database("filename.tdb")
        # ... implement test below
    ```
    """
    return pytest.mark.parametrize("load_database", [path], indirect=True)


class ConvergenceFailureSolver(Solver):
    """Solver that is guaranteed to produce a convergence failure for any composition sets or conditions passed"""
    def solve(self, composition_sets, conditions):
        spec = self.get_system_spec(composition_sets, conditions)
        self._fix_state_variables_in_compsets(composition_sets, conditions)
        state = spec.get_new_state(composition_sets)
        return SolverResult(converged=False, x=composition_sets[0].dof, chemical_potentials=np.full_like(state.chemical_potentials, np.nan))