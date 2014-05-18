import calphad.io.libtdbcpp as ctdb

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
		self._database = ctdb.DatabaseTDB()
		self.type = type
		if (filepath != ''):
			handle = open(filepath,'r')
			while True:
				line = handle.readline()
				if not line:
					break
				print(line)
			handle.close()
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
		self._database.process_command(command)
		pass
	pass