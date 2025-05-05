#include <init.hpp>
#include <preprocess.hpp>
#include <process.hpp>
#include <postprocess.hpp>
#include <bc.hpp>
#include <fileutilities.hpp>
#include <string>
#include <logger.hpp>

int main(int argc, char* argv[])
{	
	if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <name>" << std::endl;
        return 1;
    }
    
    std::string name = argv[1];

	Piro::Logger::info("Start Program");
	Piro::file_utilities::get_exec_directory();
	preprocess(name);
	solve();

	Piro::Logger::info("End Program");
	// postprocess("T");
	return 0;
}