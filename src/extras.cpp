#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/init.hpp"
#include <mach-o/dyld.h>
int writefile(std::string file_path, std::string line_to_write){
    // Create an output file stream (ofstream) object
    std::ofstream outfile(file_path, std::ios::app);  // Open the file in append mode

    // Check if the file is open
    if (outfile.is_open()) {
        // Write the line to the file
        outfile << line_to_write << std::endl;

        // Close the file stream
        outfile.close();
        std::cout << "Line written to file successfully" << std::endl;
        // std::cout << "Line written to file successfully." << std::endl;
    } else {
        std::cout << "Filed to open the file" << std::endl;
        // std::cerr << "Failed to open the file." << std::endl;
    }
    return 0;
}

int create_directory(std::string directoryname){
    std::filesystem::path dir(directoryname);

    try {
        if (std::filesystem::exists(dir)) {
            std::cout << "Directory already exists: " << dir << std::endl;
        } else {
            if (std::filesystem::create_directory(dir)) {
                std::cout << "Directory created successfully: " << dir << std::endl;
            } else {
                std::cerr << "Failed to create directory: " << dir << std::endl;
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}

int get_exec_directory(){
    char path[1024];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) == 0) {
        std::filesystem::path exe_path = path;
        current_path = exe_path.parent_path();
        //std::cout << "Executable directory: " current_path.string() << std::endl;
    } else {
        std::cerr << "Buffer size is too small; need size " << size << std::endl;
    }
    return 0;
}