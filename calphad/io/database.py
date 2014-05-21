"""The database module provides support for reading and writing data types
associated with structured thermodynamic data.

"""
import calphad.io.libtdbcpp as ctdb
from itertools import ifilterfalse, tee

class Database():
	"""
	Structured thermodynamic data.
	
	Attributes
	----------
	type : string
	    Format of the source data.
	    
	Methods
	-------
	raw_command(command)
	    Execute a raw command on the data.
	    
	"""
	def raw_command(self,command):
		"""
		Execute a raw command on the data.
		The effects depend on the specifics of the implementation.
		
		Parameters
		----------
		command : Command string
		
		Returns
		-------
		Nothing.
		    
		See Also
		--------
		Nothing.

		"""
		pass
	pass

class TDB(Database):
	"""
	Structured thermodynamic data in TDB format.
	
	Attributes
	----------
	type : {'thermocalc'}
	    Storage format of the source database.
	    
	Methods
	-------
	raw_command(command)
	    Send a raw command to the parser.
	    
	"""
	def __init__(self,filepath='',type='thermocalc'):
		"""
		Load a thermodynamic database (TDB).
		
		Parameters
		----------
		filepath : string, optional
		    Path to TDB file.
		type : {'thermocalc'}, optional
		    Storage format of the source database.
		
		Examples
		--------
		>>> emptytdb = calphad.io.database.TDB()
		>>> mytdb = calphad.io.database.TDB('feconi.tdb')
		
		"""
		self._database = ctdb.Database()
		self.type = type
		if (filepath != ''):
			self._database = ctdb.Database(filepath) # temporary
			"""
			handle = open(filepath,'r')
			lines = handle.read() # read entire file
			handle.close()
			# Take care of tabs, extra whitespace
			lines = lines.replace('\t',' ')
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
			# Separate out all PARAMETER commands; to be executed last
			commands,para_commands = partition(
				lambda cmd: cmd.upper().startswith("PARA"), 
				commands)
			map(self.raw_command,commands)
			map(self.raw_command,para_commands)
			"""
	def raw_command(self,command):
		"""
		Send a raw command to the TDB parser.
		All commands passed this way must be self-contained (i.e., terminated).
		The order in which commands are passed matters!
		
		Parameters
		----------
		command : Command string with terminator (e.g., '!' in Thermo-Calc)
		
		Returns
		-------
		Nothing.
		    
		See Also
		--------
		Nothing.
		
		Examples
		--------
		>>> mytdb = calphad.io.database.TDB()
		>>> mytdb.raw_command("ELEMENT AL FCC_A1 2.6982E+01 4.5773E+03  2.8322E+01!")
		>>> mytdb.raw_command("TYPE_DEFINITION % SEQ *!")
		>>> mytdb.raw_command("PHASE LIQUID:L %  1  1.0  !")
		>>> mytdb.raw_command("FUNCTION  GALHCP   298.0  +5481-1800*T;  6000  N !")
		>>> mytdb.raw_command("PARAMETER G(LIQUID,AL;0) 298.15 +10465.5-3.39259*T; 6000 N !")
		
		"""
		print(command)
		self._database.process_command(command)
		pass
	pass

def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = tee(iterable)
    return ifilterfalse(pred, t1), filter(pred, t2)