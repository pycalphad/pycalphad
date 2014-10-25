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
    None yet.

    Methods
    -------
    None yet.

    """
    class Phase(object):
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
        self._elements = set()
        self._species = set()
        self._phases = {}
        self._structure_dict = {} # System-local phase names to global IDs
        self._parameters = TinyDB(storage=MemoryStorage)
        self._symbols = {}
        self._references = {}
        # Note: No typedefs here (from TDB files)
        # Instead we put that information in the model_hint for _phases
    def add_element(self, element):
        """
        Add an element.

        Parameters
        ----------
        element : string
            Name of the element.

        Examples
        --------
        >>>> db = Database()
        >>>> db.add_element('Al')
        """
        self._elements.add(element.upper())
    def add_species(self, species):
        """
        Add a species.

        Parameters
        ----------
        species : string
            Name of the species.

        Examples
        --------
        >>>> db = Database()
        >>>> db.add_species('CO2')
        """
        # TODO: Verify that species def is all defined elements
        self._species.add(species.upper())
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
    def add_symbol(self, name, symbol):
        """
        Add a symbol. This is a FUNCTION in Thermo-Calc lingo.

        Parameters
        ----------
        name : string
            Name of the symbol.
        symbol : object
            Abstract representation of symbol, e.g., in SymPy format.

        Examples
        --------
        None yet.
        """
        self._symbols[name.upper()] = symbol
    def add_reference(self, name, reference):
        """
        Add a reference to a source of information.

        Parameters
        ----------
        name : string
            Unique name for the reference.
        reference : string
            A citation to a source of information.

        Examples
        --------
        None yet.
        """
        self._references[name.upper()] = reference
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
        self._phases[phase_name] = new_phase
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
            self._phases[phase_name].constituents = constituents
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
    def to_json(self):
        """
        Serialize Database to espei JSON format.
        """
        pass
    def from_json(self, input_json):
        """
        Construct Database from espei JSON format.

        Parameters
        ----------
        input_json : string
            Raw data or path to JSON file.
        """
        pass

if __name__ == "__main__":
    pass
