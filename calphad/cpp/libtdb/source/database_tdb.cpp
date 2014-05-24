/*=============================================================================
	Copyright (c) 2012-2013 Richard Otis

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/

// database_tdb.cpp -- implementation of DatabaseTDB constructor

#include "libtdb/include/libtdb_pch.hpp"
#include "libtdb/include/database_tdb.hpp"
#include "libtdb/include/parameter.hpp"
#include <iostream>
#include <fstream>
#include <boost/algorithm/string/find.hpp>

Database::DatabaseTDB::DatabaseTDB(std::string path) {
	int linecount = 0; // linecount will keep track of which line we are reading for debugging purposes
	std::fstream tdbfile;
	std::string line; // buffer for the current line
	std::vector<std::pair<std::string,int>> param_buf; // store all parameter commands with linecount, run them at the end

	RegisterCallbacks(); // register the command parser functions
	statevars.add("T",boost::spirit::utree('T')); // add Temperature as state variable
	statevars.add("P",boost::spirit::utree('P')); // add Pressure as state variable

	// initialize reserved phase type keywords
	reserved_phase_keywords["L12"] = "L12";
	reserved_phase_keywords["A1"] = "A1";
	reserved_phase_keywords["A2"] = "A2";
	reserved_phase_keywords["A3"] = "A3";
	reserved_phase_keywords["A12"] = "A12";
	reserved_phase_keywords["L12"] = "L12";
	reserved_phase_keywords["B2"] = "B2";
	reserved_phase_keywords["B32"] = "B32";
	reserved_phase_keywords["L21"] = "L21";
	reserved_phase_keywords["D019"] = "D019";
	reserved_phase_keywords["B19"] = "B19";
	reserved_phase_keywords["LAVES"] = "LAVES";
	reserved_phase_keywords["L"] = "L";

	try {
		// begin reading
		tdbfile.open(path);
		if (!tdbfile.good()) BOOST_THROW_EXCEPTION(file_read_error() << boost::errinfo_file_name(path));

		while (std::getline(tdbfile,line)) {
			++linecount;
			boost::algorithm::replace_all(line,"\t"," "); // replace tabs with spaces
			boost::algorithm::trim_all(line); // removing leading and trailing spaces, middle spaces truncated to one space
			if (boost::algorithm::starts_with(line,"$")) continue; // skip lines that are comments
			if (line.empty()) continue; // skip lines that are empty

			int templinecount = linecount; // store linecount prior to iteration for better error reporting

			auto line_iters = boost::algorithm::find_first(line, "!");
			while (line_iters.begin() == line.end()) { // the current command doesn't terminate yet, keep reading
				std::string buf;
				while (buf.begin() == buf.end()) {
					std::getline(tdbfile,buf); // move until we find a non-empty line
					if (tdbfile.fail()) {
						// current command did not terminate, but there's no more data
						BOOST_THROW_EXCEPTION(parse_error() << str_errinfo("Command did not terminate") << boost::errinfo_at_line(templinecount));
					}
					boost::algorithm::trim_all(buf);
					++linecount;
				}
				line = line + " " + buf;
				line_iters = boost::algorithm::find_first(line, "!");
			}
			boost::algorithm::trim_all(line); // trim again in case we had a multi-line command that added spaces
			try {
				if (boost::algorithm::starts_with(line,"PARA")) 
					param_buf.push_back(std::pair<std::string,int>(line,templinecount)); // add to buffer
				else proc_command(line); // process the command and add the data to this Database object
			}
			catch (parse_error &e) {
				e << boost::errinfo_at_line(templinecount); // add line number to exception
				throw; // push the error up the call stack
			}
		}

		// Execute all PARAMETER commands in the buffer
		for (auto i = param_buf.begin(); i != param_buf.end(); ++i) {
			try {
				proc_command(i->first); // execute a PARAMETER command
			}
			catch (parse_error &e) {
				e << boost::errinfo_at_line(i->second); // add line number to exception
				throw; // push the error up the call stack
			}
		}
		// if there is a file error and we didn't finish reading, throw an exception
		if (!tdbfile.good() && !tdbfile.eof()) BOOST_THROW_EXCEPTION(file_read_error() << boost::errinfo_file_name(path));
	}
	catch (parse_error &e) {
		// primary error handling for parse errors
		// in the future this could use errno's to intelligently handle these
		std::string specific_info, err_msg; // error message strings
		int linenum; // line number of error
		if (std::string const * mi = boost::get_error_info<specific_errinfo>(e) ) {
			specific_info = *mi;
		}
		if (std::string const * mi = boost::get_error_info<str_errinfo>(e) ) {
			err_msg = *mi;
		}
		if (int const * mi = boost::get_error_info<boost::errinfo_at_line>(e) ) {
			linenum = *mi;
		}
		std::cerr << "Exception: " << err_msg << " on line " << linenum << std::endl;
		std::cerr << "Reason: " << specific_info << std::endl;
		//std::cerr << std::endl << std::endl << diagnostic_information(e);
                throw;
	}
	catch (file_read_error &e) {
		// 'path' is in scope here, but just for safety we'll read it from the exception object
		std::string fname;
		if (std::string const * mi = boost::get_error_info<boost::errinfo_file_name>(e) ) {
			fname = *mi;
		}
		std::cerr << "Cannot read from \"" << fname << "\"" << std::endl;
                throw;
	}
}

//
// Forwarding functions for Database class 
//

// initialize the opaque pointer pImpl with the implementation DatabaseTDB
Database::Database(std::string s): pImpl (new DatabaseTDB(s))
{
}

Database::Database(): pImpl (new DatabaseTDB()) { }

void Database::set_info(std::string infostring) {
	pImpl->set_info(infostring);
}
void Database::proc_command(std::string cmd) {
    pImpl->proc_command(cmd);
}
std::string Database::get_info() const {
	return pImpl->get_info();
}
Element Database::get_element(std::string s) const {
	return pImpl->get_element(s);
}
Species_Collection Database::get_all_species() const {
	return pImpl->get_all_species();
}
Phase_Collection::const_iterator Database::get_phase_iterator() const {
	return pImpl->get_phase_iterator();
}
Phase_Collection::const_iterator Database::get_phase_iterator_end() const {
	return pImpl->get_phase_iterator_end();
}
Phase_Collection Database::get_phases() const {
        return pImpl->get_phases();
}
parameter_set Database::get_parameter_set() const {
	return pImpl->get_parameter_set();
}
