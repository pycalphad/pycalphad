// database.h -- header file for pImpl-style encapsulation
//               of the Database class
// The actual Database class is defined in libTDB (or elsewhere)
// but has lots of dependencies. This saves us lots of compile time.

#ifndef INCLUDED_DATABASE_IMPL
#define INCLUDED_DATABASE_IMPL

#include <memory>
#include <string>
#include "structure.h"

class Database {
	class DatabaseTDB;
	std::shared_ptr<DatabaseTDB> pImpl;
public:
	Database(std::string);
	void set_info(std::string &infostring); // set infostring for the database
	std::string get_info() const; // get infostring for database
	(::Element) get_element(std::string s) const;
	Species_Collection get_all_species() const;
	Phase_Collection::const_iterator get_phase_iterator() const;
	Phase_Collection::const_iterator get_phase_iterator_end() const;
};

#endif