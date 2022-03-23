from importlib_resources import files
import pytest
import pycalphad.tests.databases
from pycalphad.io.database import Database

@pytest.fixture(scope="session")
def load_database(request):
    """
    Helper fixture to load a database (parameterized by the value of `request`).
    """
    db = Database(str(files(pycalphad.tests.databases).joinpath(request.param)))
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
