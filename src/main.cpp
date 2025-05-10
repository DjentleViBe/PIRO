#include <init.hpp>
#include <preprocess.hpp>
#include <process.hpp>
#include <postprocess.hpp>
#include <bc.hpp>
#include <fileutilities.hpp>
#include <string>
#include <logger.hpp>
#include <openclutilities.hpp>
int compile;

int main(int argc, char* argv[])
{	
	if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <name>" << std::endl;
        return 1;
    }
    
    std::string name = argv[1];
	compile = std::stoi(argv[2]);

	Piro::logger::info("Start Program");
	Piro::file_utilities::get_exec_directory();
	Piro::preprocess(name);
	if(compile == 0){
		Piro::solve();
	}

	Piro::logger::info("End Program");
	// postprocess("T");
	return 0;
}