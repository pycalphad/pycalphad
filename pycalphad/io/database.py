"""
The database module provides support for reading and writing data types
associated with structured thermodynamic/kinetic data.
"""
from io import StringIO
from tinydb import TinyDB
from tinydb.storages import MemoryStorage
from datetime import datetime
from collections import namedtuple
import os
from pycalphad.variables import Species
from pycalphad.core.cache import fhash
from pycalphad.core.utils import recursive_tuplify


class DatabaseExportError(Exception):
    """Raised when a database cannot be written."""
    pass


class Phase(object): #pylint: disable=R0903
    """
    Phase in the database.

    Attributes
    ----------
    name : string
        System-local name of the phase.
    constituents : tuple of frozenset
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
    def __eq__(self, other):
        if type(self) == type(other):
            return self.__dict__ == other.__dict__
        else:
            return False
    def __ne__(self, other):
        return not self.__eq__(other)
    def __repr__(self):
        return 'Phase({0!r})'.format(self.__dict__)
    def __hash__(self):
        return hash((self.name, self.constituents, tuple(self.sublattices),
                     tuple(sorted(recursive_tuplify(self.model_hints.items())))))

DatabaseFormat = namedtuple('DatabaseFormat', ['read', 'write'])
format_registry = {}


class Database(object): #pylint: disable=R0902
    """
    Structured thermodynamic and/or kinetic data.

    Attributes
    ----------
    elements : set
        Set of elements in database.
    species : set
        Set of species in database.
    phases : dict
        Phase objects indexed by their system-local name.
    symbols : dict
        SymEngine objects indexed by their name (FUNCTIONs in Thermo-Calc).
    references : dict
        Reference objects indexed by their system-local identifier.

    Examples
    --------
    >>> mydb = Database(open('crfeni_mie.tdb'))
    >>> mydb = Database('crfeni_mie.tdb')
    >>> f = StringIO(u'$a complete TDB file as a string\\n')
    >>> mydb = Database(f)
    """
    def __new__(cls, *args):
        if len(args) == 0:
            obj = super(Database, cls).__new__(cls, *args)
            # Should elements be rolled into a special case of species?
            obj.elements = set()
            obj.species = set()
            obj.phases = {}
            obj.refstates = {}
            obj._structure_dict = {} # System-local phase names to global IDs
            obj._parameters = TinyDB(storage=MemoryStorage)
            obj._parameter_queue = []
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
            except (AttributeError, TypeError):
                pass
            if hasattr(fname, 'read'):
                # File descriptor
                return cls.from_file(fname, fmt=fmt)
            elif fname.find('\n') == -1:
                # Single-line string; it's probably a filename
                return cls.from_file(fname, fmt=fmt)
            else:
                # Newlines found: probably a full database string
                return cls.from_string(fname, fmt=fmt)
        else:
            raise ValueError('Invalid number of parameters: '+len(args))

    def __hash__(self):
        return fhash(self.__dict__)


    def __getstate__(self):
        pickle_dict = {}
        for key, value in self.__dict__.items():
            if key == '_parameters':
                pickle_dict[key] = value.all()
            else:
                pickle_dict[key] = value
        return pickle_dict

    def __setstate__(self, state):
        for key, value in state.items():
            if key == '_parameters':
                self._parameters = TinyDB(storage=MemoryStorage)
                self._parameters.insert_multiple(value)
            else:
                setattr(self, key, value)

    def __deepcopy__(self, memo):
        copy = type(self)()
        memo[id(self)] = copy
        for key, value in self.__dict__.items():
            if key == '_parameters':
                copy._parameters = TinyDB(storage=MemoryStorage)
                copy._parameters.insert_multiple(value.all())
            else:
                setattr(copy, key, value)
        return copy

    @staticmethod
    def register_format(fmt, read=None, write=None):
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
            except (AttributeError, TypeError):
                # fname isn't actually a path, so we don't know the correct format
                raise ValueError('\'fmt\' keyword argument must be specified when passing a file descriptor.')
            if '.' in ext and ext[1:].lower() in format_registry:
                fmt = ext[1:].lower()
        else:
            fmt = fmt.lower()
        if fmt not in format_registry or format_registry[fmt].read is None:
            supported_reads = [key for key, value in format_registry.items() if value.read is not None]
            raise NotImplementedError('Unsupported read format \'{0}\'. Supported formats: {1}'.format(fmt,
                                                                                                        supported_reads))
        # Is it a file descriptor?
        if hasattr(fname, 'read'):
            fd = fname
            need_to_close = False
        else:
            # It's not file-like, so it's probably a filename
            need_to_close = True
            fd = open(fname, mode='r')
        try:
            dbf = Database()
            format_registry[fmt.lower()].read(dbf, fd)
        finally:
            # Close file descriptors created in this routine
            # Otherwise that's left up to the calling function
            if need_to_close:
                fd.close()

        return dbf

    @classmethod
    def from_string(cls, data, **kwargs):
        """
        Returns Database from a string in the specified format.
        This function is a wrapper for calling `from_file` with StringIO.

        Parameters
        ----------
        data : str
            Raw database string in the specified format.
        kwargs : optional
            See keyword arguments for `from_file`.

        Returns
        -------
        dbf : Database
        """
        return cls.from_file(StringIO(data), **kwargs)

    def to_file(self, fname, fmt=None, if_exists='raise', **write_kwargs):
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
            This argument is ignored if 'fname' is file-like.
        write_kwargs : optional
            Keyword arguments to pass to write function.

        Examples
        --------
        None yet.
        """
        if fmt is None:
            # Attempt to auto-detect the correct format based on the file extension
            try:
                path, ext = os.path.splitext(fname)
            except (AttributeError, TypeError):
                # fname isn't actually a path, so we don't know the correct format
                raise ValueError('\'fmt\' keyword argument must be specified when passing a file descriptor.')
            if '.' in ext and ext[1:].lower() in format_registry:
                fmt = ext[1:].lower()
        else:
            fmt = fmt.lower()
        if fmt not in format_registry or format_registry[fmt].write is None:
            supported_writes = [key for key, value in format_registry.items() if value.write is not None]
            raise NotImplementedError('Unsupported write format \'{0}\'. Supported formats: {1}'.format(fmt,
                                                                                                        supported_writes))
        # Is this a file descriptor?
        if hasattr(fname, 'write'):
            format_registry[fmt].write(self, fname, **write_kwargs)
        else:
            if os.path.exists(fname) and if_exists != 'overwrite':
                if if_exists == 'rename':
                    writetime = datetime.now()
                    fname = os.path.splitext(fname)
                    fname = fname[0] + "." + writetime.strftime("%Y-%m-%d-%H-%M") + fname[1]
                else:
                    # equivalent to 'raise'
                    raise FileExistsError('File {} already exists'.format(fname))
            with open(fname, mode='w') as fd:
                format_registry[fmt].write(self, fd, **write_kwargs)

    def to_string(self, **kwargs):
        """
        Returns Database as a string.
        This function is a wrapper for calling `to_file` with StringIO.

        Parameters
        ----------
        kwargs : optional
            See keyword arguments for `to_file`.

        Returns
        -------
        result : str
        """
        result = StringIO()
        self.to_file(result, **kwargs)
        return result.getvalue()

    def __str__(self):
        result = 'Elements: {0}\n'.format(sorted(self.elements))
        result += 'Species: {0}\n'.format(sorted(self.species, key=lambda s: s.name))
        for name, phase in sorted(self.phases.items()):
            result += str(phase)+'\n'
        result += '{0} symbols in database\n'.format(len(self.symbols))
        result += '{0} parameters in database\n'.format(len(self._parameters))
        return result

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) != type(other):
            return False
        elif sorted(self.__dict__.keys()) != sorted(other.__dict__.keys()):
            return False
        else:
            def param_sort_key(x):
                return x['phase_name'], x['parameter_type'], x['constituent_array'], \
                       x['parameter_order'], x['diffusing_species']
            for key in self.__dict__.keys():
                if key == '_parameters':
                    # Special handling for TinyDB objects
                    if len(self._parameters.all()) != len(other._parameters.all()):
                        return False
                    self_params = sorted(self._parameters.all(), key=param_sort_key)
                    other_params = sorted(other._parameters.all(), key=param_sort_key)
                    if self_params != other_params:
                        return False
                elif self.__dict__[key] != other.__dict__[key]:
                    return False
            return True

    def __ne__(self, other):
        return not self.__eq__(other)

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
            Abstract representation of symbol, e.g., in SymEngine format.

        Examples
        --------
        None yet.
        """
        self._structure_dict[local_name] = global_name

    def add_parameter(
        self, param_type, phase_name, constituent_array, param_order, param, ref=None,
        diffusing_species=None, force_insert=True, **kwargs,
        ):
        """
        Add a parameter.

        Parameters
        ----------
        param_type : str
            Type name of the parameter, e.g., G, L, BMAGN.
        phase_name : str
            Name of the phase.
        constituent_array : list
            Configuration of the sublattices (elements and/or species).
        param_order : int
            Polynomial order of the parameter.
        param : object
            Abstract representation of the parameter, e.g., in SymEngine format.
        ref : str, optional
            Reference for the parameter.
        diffusing_species : str, optional
            (If kinetic parameter) Diffusing species for this parameter.
        force_insert : bool, optional
            If True, inserts into the database immediately. False is a delayed insert (for performance).
        kwargs : Any
            Additional metadata to insert into the parameter dictionary

        Examples
        --------
        None yet.
        """
        species_dict = {s.name: s for s in self.species}
        new_parameter = {
            'phase_name': phase_name,
            'constituent_array': tuple(tuple(species_dict.get(s.upper(), Species(s)) for s in xs) for xs in constituent_array),  # must be hashable type
            'parameter_type': param_type,
            'parameter_order': param_order,
            'parameter': param,
            'diffusing_species': Species(diffusing_species),
            'reference': ref
        }
        new_parameter.update(kwargs)
        if force_insert:
            self._parameters.insert(new_parameter)
        else:
            self._parameter_queue.append(new_parameter)

    def add_phase(self, phase_name, model_hints, sublattices):
        """
        Add a phase.

        Parameters
        ----------
        phase_name : string
            System-local name of the phase.
        model_hints : dict
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
        # Need to convert from ParseResults or else equality testing will break
        new_phase.sublattices = tuple(sublattices)
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
        species_dict = {s.name: s for s in self.species}
        try:
            # Need to convert constituents from ParseResults
            # Otherwise equality testing will be broken
            self.phases[phase_name].constituents = tuple([frozenset([species_dict[s.upper()] for s in xs]) for xs in constituents])
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

    def process_parameter_queue(self):
        """
        Process the queue of parameters so they are added to the TinyDB in one transaction.
        This avoids repeated (expensive) calls to insert().
        """
        result = self._parameters.insert_multiple(self._parameter_queue)
        self._parameter_queue = []
        return result
