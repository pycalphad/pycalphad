#include "stdafx.h"
#include "database_tdb.h"


void Database::DatabaseTDB::proc_command(std::string &cmdstring) {
	auto cmdrange = boost::find_first(cmdstring," ");
	std::string cmd(cmdstring.begin(),cmdrange.end()); // current command name
	boost::trim_right(cmd); // remove the trailing space from the command name
	boost::to_upper(cmd); // force command to be uppercase
	//std::cout << cmd << " " << std::string(cmdrange.end(),cmdstring.end()) << std::endl;
	if (parser_map.find(cmd) != parser_map.end()) {
		std::string args(cmdrange.end(),cmdstring.end());
		boost::trim_right_if(args, boost::is_any_of(" !")); // trim away end token and any trailing spaces (should all be gone)
		try {
			(this->*parser_map[cmd])(args); // execute parser callback function
		}
		catch (parse_error &e) {
			e << str_errinfo("Parse error for command \""+ cmd +"\"");
			throw; // push exception up the call stack
		}
	}
	else {
		BOOST_THROW_EXCEPTION(parse_error() << str_errinfo("Unknown command") << specific_errinfo(cmd));
	}
}