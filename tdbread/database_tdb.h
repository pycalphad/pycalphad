#ifndef INCLUDED_DATABASE_TDB
#define INCLUDED_DATABASE_TDB

#include "warning_disable.h"
#include <map>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/trim_all.hpp>
#include <boost/spirit/home/support/utree.hpp>
#include <boost/spirit/include/qi_symbols.hpp>

#include "structure.h"
#include "exceptions.h"

class Database {
private:
	std::string name;
	std::string info;
	Element_Collection elements; // all pure elements from this database
	Phase_Collection phases; // all phases
	Species_Collection myspecies; // all species
	boost::spirit::qi::symbols<char, boost::spirit::utree> macros; // all of the macros (FUNCTIONs in Thermo-Calc lingo)
	boost::spirit::qi::symbols<char, boost::spirit::utree> statevars; // all valid state variables
	std::map<std::string,std::string> reserved_phase_keywords; // reserved phase suffixes: L12, A2, LAVES, etc.

	typedef void (Database:: *ParserCallback)(std::string &);
	std::map<std::string, ParserCallback>  parser_map; // maps commands from input database to a parser function

	void proc_command(std::string &); // internal parser function
	void RegisterParserCallback(std::string cmdname, ParserCallback pcb) { parser_map[cmdname] = pcb; };
	void RegisterCallbacks() { // initialize the parser map
			// Parser callback functions get a string containing everything after the first command name
			// spaces and '!' trimmed
			RegisterParserCallback("DATABASE_INFO", &Database::Database_Info);
			RegisterParserCallback("ELEMENT", &Database::Element);
			RegisterParserCallback("SPECIES", &Database::Species);
			RegisterParserCallback("PHASE", &Database::Phase);
			RegisterParserCallback("CONSTITUENT", &Database::Constituent);
			RegisterParserCallback("ADD_CONSTITUENT", &Database::Constituent);
			RegisterParserCallback("FUNCTION", &Database::Function);
			RegisterParserCallback("FUN", &Database::Function);
			RegisterParserCallback("PARAMETER", &Database::Parameter);
			RegisterParserCallback("PARA", &Database::Parameter);

			RegisterParserCallback("TYPE_DEFINITION", &Database::Unsupported_Command);
			RegisterParserCallback("TYPE_DEF", &Database::Unsupported_Command);
			RegisterParserCallback("DEFINE_SYSTEM_DEFAULT", &Database::Unsupported_Command);
			RegisterParserCallback("DEFAULT_COMMAND", &Database::Unsupported_Command);
	};
	bool check_formula_validity(chemical_formula); // checks that chemical formula has all defined Elements

	// parser callback functions
	void Unsupported_Command(std::string &) { };
	void Database_Info(std::string &);
	void Element(std::string &);
	void Species(std::string &);
	void Phase(std::string &);
	void Constituent(std::string &);
	void Function(std::string &);
	void Parameter(std::string &);

	void Species(::Element); // non-callback, internal version for pure elements
public:
	Database() { RegisterCallbacks(); };
	Database(std::string);

	void set_info(std::string &infostring) { info = infostring; }; // set infostring for the database
	std::string get_info() { return info; } // get infostring for database
	(::Element) get_element(std::string s) { return elements[s]; }
	Species_Collection get_all_species() { return myspecies; }
	Phase_Collection::const_iterator get_phase_iterator() const { return phases.cbegin(); }
	Phase_Collection::const_iterator get_phase_iterator_end() const { return phases.cend(); }
};

#endif