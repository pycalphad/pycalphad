"""The database module provides support for reading and writing data types
associated with structured thermodynamic/kinetic data.
"""
try:
    set
except NameError:
    from sets import Set as set #pylint: disable=W0622
from tinydb import TinyDB
from tinydb.storages import MemoryStorage

class Database(object):
    """
    Structured thermodynamic and/or kinetic data.
    Databases are usually filled by Parsers and read by Models.

    Attributes
    ----------
    elements : list
        List of elements in database.
    species : list
        List of species in database.
    phases : dict
        Phase objects indexed by their system-local name.
    symbols : dict
        SymPy objects indexed by their name (FUNCTIONs in Thermo-Calc).
    references : dict
        Reference objects indexed by their system-local identifier.

    Methods
    -------
    None yet.

    """
    class Phase(object): #pylint: disable=R0903
        """
        Phase in the database.

        Attributes
        ----------
        name : string
            System-local name of the phase.
        constituents : list of lists
            Possible sublattice constituents (elements and/or species).
        sublattices : list
            Site ratios of sublattices.
        model_hints : list
            Structured "hints" for a Model trying to read this phase.
            Hints for major constituents and typedefs (Thermo-Calc) go here.
        """
        def __init__(self):
            self.name = None
            self.constituents = None
            self.sublattices = []
            self.model_hints = []
    def __init__(self):
        self.elements = set()
        self.species = set()
        self.phases = {}
        self._structure_dict = {} # System-local phase names to global IDs
        self._parameters = TinyDB(storage=MemoryStorage)
        self.symbols = {}
        self.references = {}
        # Note: No typedefs here (from TDB files)
        # Instead we put that information in the model_hint for phases
    def add_structure_entry(self, local_name, global_name):
        """
        Define a relation between the system-local name of a phase and a
        "global" identifier. This is used to link crystallographically
        similar phases known by different colloquial names.

        Parameters
        ----------
        local_name : string
            System-local name of the phase.
        global_name : object
            Abstract representation of symbol, e.g., in SymPy format.

        Examples
        --------
        None yet.
        """
        self._structure_dict[local_name] = global_name
    def add_parameter(self, param_type, phase_name, #pylint: disable=R0913
                      constituent_array, param_order,
                      param, ref=None):
        """
        Add a parameter.

        Parameters
        ----------
        param_type : str
            Type name of the parameter, e.g., G, L, BMAGN.
        phase_name : string
            Name of the phase.
        constituent_array : list
            Configuration of the sublattices (elements and/or species).
        symbol : object
            Abstract representation of the parameter, e.g., in SymPy format.

        Examples
        --------
        None yet.
        """
        new_parameter = {
            'phase_name': phase_name,
            'constituent_array': constituent_array,
            'parameter_type': param_type,
            'parameter_order': param_order,
            'parameter': param,
            'reference': ref
        }
        param_id = self._parameters.insert(new_parameter)
        return param_id
    def add_phase(self, phase_name, model_hints, sublattices):
        """
        Add a phase.

        Parameters
        ----------
        phase_name : string
            System-local name of the phase.
        model_hints : list
            Structured "hints" for a Model trying to read this phase.
            Hints for major constituents and typedefs (Thermo-Calc) go here.
        sublattices : list
            Site ratios of sublattices.

        Examples
        --------
        None yet.
        """
        new_phase = Database.Phase()
        new_phase.name = phase_name
        new_phase.sublattices = sublattices
        new_phase.model_hints = model_hints
        self.phases[phase_name] = new_phase
    def add_phase_constituents(self, phase_name, constituents):
        """
        Add a phase.

        Parameters
        ----------
        phase_name : string
            System-local name of the phase.
        constituents : list
            Possible phase constituents (elements and/or species).

        Examples
        --------
        None yet.
        """
        try:
            self.phases[phase_name].constituents = constituents
        except KeyError:
            print("Undefined phase "+phase_name)
            raise
    def search(self, query):
        """
        Search for parameters matching the specified query.

        Parameters
        ----------
        query : object
            Structured database query in TinyDB format.

        Examples
        --------
        >>>> from tinydb import where
        >>>> db = Database()
        >>>> eid = db.add_parameter(...) #TODO
        >>>> db.search(where('eid') == eid)
        """
        return self._parameters.search(query)

if __name__ == "__main__":
    pass
