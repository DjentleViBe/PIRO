#include <init.hpp>
#include <preprocess.hpp>
#include <solve.hpp>
#include <postprocess.hpp>
#include <bc.hpp>
#include <fileutilities.hpp>
#include <string>

int main(int argc, char* argv[])
{	
	if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <name>" << std::endl;
        return 1;
    }
    
    std::string name = argv[1];

	Piro::FileUtilities::get_exec_directory();
	preprocess(name);
	solve();
	// postprocess("T");
	return 0;
}