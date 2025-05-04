#include "../dependencies/include/init.hpp"
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/postprocess.hpp"
#include "../dependencies/include/bc.hpp"
#include "../dependencies/include/fileutilities.hpp"
#include <string>

int main(int argc, char* argv[])
{	
	if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <name>" << std::endl;
        return 1;
    }
    
    std::string name = argv[1];

	get_exec_directory();
	preprocess(name);
	solve();
	// postprocess("T");
	return 0;
}