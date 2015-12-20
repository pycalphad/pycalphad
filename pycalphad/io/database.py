"""The database module provides support for reading and writing data types
associated with structured thermodynamic/kinetic data.
"""
from tinydb import TinyDB
from tinydb.storages import MemoryStorage
from datetime import datetime
from collections import namedtuple
import os
try:
    # Python 2
    from StringIO import StringIO
except ImportError:
    # Python 3
    from io import StringIO


def _to_tuple(lst):
    "Convert nested list to nested tuple. Source: Martijn Pieters on StackOverflow"
    return tuple(_to_tuple(i) if isinstance(i, list) else i for i in lst)

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
    model_hints : dict
        Structured "hints" for a Model trying to read this phase.
        Hints for major constituents and typedefs (Thermo-Calc) go here.
    """
    def __init__(self):
        self.name = None
        self.constituents = None
        self.sublattices = []
        self.model_hints = {}
    def __repr__(self):
        return 'Phase({0!r})'.format(self.__dict__)

DatabaseFormat = namedtuple('DatabaseFormat', ['read', 'write'])
format_registry = {}


class Database(object): #pylint: disable=R0902
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
    def __new__(cls, *args):
        """
        Construct a Database object.

        Parameters
        ----------
        args: file descriptor or raw data, optional
              File to load.

        Examples
        --------
        >>> mydb = Database(open('crfeni_mie.tdb'))
        >>> mydb = Database('crfeni_mie.tdb')
        >>> f = StringIO(u'$a complete TDB file as a string\n')
        >>> mydb = Database(f)
        """
        if len(args) == 0:
            obj = super(Database, cls).__new__(cls, *args)
            # Should elements be rolled into a special case of species?
            obj.elements = set()
            obj.species = set()
            obj.phases = {}
            obj.typedefs = {}
            obj._structure_dict = {} # System-local phase names to global IDs
            obj._parameters = TinyDB(storage=MemoryStorage)
            obj.symbols = {}
            obj.references = {}
            # Note: No public typedefs here (from TDB files)
            # Instead we put that information in the model_hint for phases
            return obj
        elif len(args) == 1:
            fname = args[0]
            # Backwards compatibility: assume TDB by default
            fmt = 'tdb'
            # Attempt to auto-detect the correct format based on the file extension
            try:
                path, ext = os.path.splitext(fname)
                if '.' in ext and ext[1:].lower() in format_registry:
                    fmt = ext[1:].lower()
            except AttributeError:
                pass
            return cls.from_file(fname, fmt=fmt)
        else:
            raise ValueError('Invalid number of parameters: '+len(args))

    @staticmethod
    def register(fmt, read=None, write=None):
        """
        Add support for reading and/or writing the specified format.

        Parameters
        ----------
        fmt: str
            Format.
        read : callable, optional
            Read function with arguments (Database, file_descriptor)
        write : callable, optional
            Write function with arguments (Database, file_descriptor)

        Examples
        --------
        None yet.
        """
        format_registry[fmt.lower()] = DatabaseFormat(read=read, write=write)

    @staticmethod
    def from_file(fname, fmt=None):
        """
        Create a Database from a file.

        Parameters
        ----------
        fname: str or file-like
            File name/descriptor to read.
        fmt : str, optional
            File format. If not specified, an attempt at auto-detection is made.

        Returns
        -------
        dbf : Database
            Database from file.

        Examples
        --------
        None yet.
        """
        if fmt is None:
            # Attempt to auto-detect the correct format based on the file extension
            try:
                path, ext = os.path.splitext(fname)
            except AttributeError:
                # fname isn't actually a path, so we don't know the correct format
                raise ValueError('\'fmt\' keyword argument must be specified when passing a file descriptor.')
            if '.' in ext and ext[1:].lower() in format_registry:
                fmt = ext[1:].lower()
        else:
            fmt = fmt.lower()
        if fmt not in format_registry or format_registry[fmt].read is None:
            supported_reads = [i for i in format_registry.keys() if i.read is not None]
            raise NotImplementedError('Unsupported read format \'{0}\'. Supported formats: {1}'.format(fmt,
                                                                                                        supported_reads))
        # First, let's try to treat it like it's a file descriptor
        try:
            fname.read
            fd = fname
            need_to_close = False
        except AttributeError:
            # It's not file-like, so it's probably string-like
            # The question is if it's a filename or the whole raw database
            # Solution is to check for newlines
            need_to_close = True
            if fname.find('\n') == -1:
                # Single-line; it's probably a filename
                fd = open(fname, mode='r')
            else:
                # Newlines found: probably a full database string
                fd = StringIO(fname)
        try:
            dbf = Database()
            format_registry[fmt.lower()].read(dbf, fd)
        finally:
            # Close file descriptors created in this routine
            # Otherwise that's left up to the calling function
            if need_to_close:
                fd.close()

        return dbf

    def to_file(self, fname, fmt=None, if_exists='raise'):
        """
        Write the Database to a file.

        Parameters
        ----------
        fname: str or file-like
            File name/descriptor to write.
        fmt : str, optional
            File format. If not specified, an attempt at auto-detection is made.
        if_exists : string, optional ['raise', 'rename', 'overwrite']
            Strategy if 'fname' already exists.
            The 'raise' option (default) will raise a FileExistsError.
            The 'rename' option will append the date/time to the filename.
            The 'overwrite' option will overwrite the file.
            This option is ignored if 'fname' is file-like.

        Examples
        --------
        None yet.
        """
        if fmt is None:
            # Attempt to auto-detect the correct format based on the file extension
            try:
                path, ext = os.path.splitext(fname)
            except AttributeError:
                # fname isn't actually a path, so we don't know the correct format
                raise ValueError('\'fmt\' keyword argument must be specified when passing a file descriptor.')
            if '.' in ext and ext[1:].lower() in format_registry:
                fmt = ext[1:].lower()
        else:
            fmt = fmt.lower()
        if fmt not in format_registry or format_registry[fmt].write is None:
            supported_writes = [i for i in format_registry.keys() if i.write is not None]
            raise NotImplementedError('Unsupported write format \'{0}\'. Supported formats: {1}'.format(fmt,
                                                                                                        supported_writes))
        # Is this a file descriptor?
        if hasattr(fname, 'write'):
            format_registry[fmt].write(self, fname)
        else:
            if os.path.exists(fname) and if_exists != 'overwrite':
                if if_exists == 'raise':
                    raise FileExistsError('File {} already exists'.format(fname))
                elif if_exists == 'rename':
                    writetime = datetime.now()
                    fname = os.path.splitext(fname)
                    fname = fname[0] + "." + writetime.strftime("%Y-%m-%d-%H-%M") + fname[1]
            with open(fname, mode='w') as fd:
                format_registry[fmt].write(self, fd)

    def __str__(self):
        result = 'Elements: {0}\n'.format(sorted(self.elements))
        result += 'Species: {0}\n'.format(sorted(self.species))
        for symbol, info in sorted(self.typedefs.items()):
            result += 'Type Definition \'{0}\': {1}\n'.format(symbol, info)
        for name, phase in sorted(self.phases.items()):
            result += str(phase)+'\n'
        result += '{0} symbols in database\n'.format(len(self.symbols))
        result += '{0} parameters in database\n'.format(len(self._parameters))
        return result

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
            'constituent_array': _to_tuple(constituent_array),  # must be hashable type
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
        new_phase = Phase()
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
