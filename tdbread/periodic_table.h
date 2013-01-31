#include <string>
#include <map>

// the element_data class contains purely chemical information about the elements
class element_data {
private:
	std::string ele_sym; // symbol of the element
	std::string fullname; // name of the element
	int atomic_number;
public:
	element_data() { };
	element_data(std::string sym, std::string name, int atno) {
		ele_sym = sym;
		fullname = name;
		atomic_number = atno;
	}
	std::string name() { return fullname; }
	std::string symbol() { return ele_sym; }
	int atno() { return atomic_number; }
};

// the periodic_table_elements map contains the periodic table
std::map<std::string,element_data> create_periodic_table(); // defined in periodic_table.cpp
extern std::map<std::string,element_data> periodic_table_elements; // defined in database_tdb.cpp