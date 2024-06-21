#include <iostream>
#include <filesystem>
#include "../dependencies/include/init.hpp"
#include "../dependencies/include/extras.hpp"

std::filesystem::path current_path;

void init(){
    /*try {
        // Get the current path
        current_path = std::filesystem::current_path();
        
        // Print the current path
        std::cout << "Current working directory: " << current_path << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }*/
    get_exec_directory();
    create_directory(current_path.string() + "/" + "mesh");
    create_directory(current_path.string() + "/" + "mesh/level");

}
