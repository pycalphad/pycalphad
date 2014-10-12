"""The database module provides support for reading and writing data types
associated with structured thermodynamic/kinetic data.
"""
from sets import Set
from tinydb import TinyDB

class Database:
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
    def __init__(self):
        self._elements = Set()
        self._species = Set()
        self._phases = {}
        self._structure_dict = {} # System-local phase names to global IDs
        self._parameters = TinyDB()
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
        if not isinstance(element, str):
            raise TypeError
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
        if not isinstance(species, str):
            raise TypeError
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
    def add_parameter(self, phase_name, constituent_array, param_type,
                      param_order, param, ref=None):
        """
        Add a parameter.

        Parameters
        ----------
        name : string
            Name of the symbol.
        symbol : object
            Abstract representation of the parameter, e.g., in SymPy format.

        Examples
        --------
        None yet.
        """
        if not isinstance(phase_name, str):
            raise TypeError
        if not isinstance(param_order, int):
            raise TypeError
        if not isinstance(param_type, str):
            raise TypeError
        if not (isinstance(ref, str) or isinstance(ref, None)):
            raise TypeError
        if not isinstance(constituent_array, list):
            raise TypeError
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
    def add_phase(self, phase_name, constituents, model_hints):
        """
        Add a phase.

        Parameters
        ----------
        phase_name : string
            System-local name of the phase.
        constituents : list
            Possible phase constituents (elements and/or species).
        model_hints : list
            Structured "hints" for a Model trying to read this phase.
            Hints for major constituents and typedefs (Thermo-Calc) go here.

        Examples
        --------
        None yet.
        """
        raise NotImplementedError
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
        raise NotImplementedError
    def to_tdb(self):
        """
        Serialize Database to Thermo-Calc TDB format.
        """
        raise NotImplementedError
    def from_json(self, input_json):
        """
        Construct Database from espei JSON format.

        Parameters
        ----------
        input_json : string
            Raw data or path to JSON file.
        """
        raise NotImplementedError
    def from_tdb(self, input_data):
        """
        Construct Database from Thermo-Calc TDB format.

        Parameters
        ----------
        input_data : string
            Raw data or path to TDB file.
        """
        if input_data.find('\n') == -1:
            # This is probably a filename rather than a raw TDB
            raw_data = None
            with open(input_data, 'r') as data:
                raw_data = data.read()
            tdbread(self, raw_data)
        else:
            tdbread(self, input_data)

from itertools import tee

def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    tee1, tee2 = tee(iterable)
    return [i for i in tee1 if not pred(i)], [i for i in tee2 if pred(i)]

def tdbread(targetdb, lines):
    """
    Parse a TDB file into a pycalphad Database object.

    Parameters
    ----------
    targetdb : Database
        A pycalphad Database.
    lines : string
        A raw TDB file.
    """
    lines = lines.replace('\t', ' ')
    lines = lines.strip()
    # Split the string by newlines
    splitlines = lines.split('\n')
    # Remove extra whitespace inside line
    splitlines = [' '.join(k.split()) for k in splitlines]
    # Remove comments
    splitlines = [k for k in splitlines if not k.startswith("$")]
    # Combine everything back together
    lines = ' '.join(splitlines)
    # Now split by the command delimeter
    commands = lines.split('!')
    # Filter out comments one more time
    # It's possible they were at the end of a command
    commands = [k for k in commands if not k.startswith("$")]
    # Separate out all PARAMETER commands; to be handled last
    commands, para_commands = partition(
        lambda cmd: cmd.upper().startswith("PARA"),
        commands)
    # Separate out all FUNCTION commands
    commands, func_commands = partition(
        lambda cmd: cmd.upper().startswith("FUNC"),
        commands)

if __name__ == "__main__":
    pass
