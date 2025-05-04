#include <filesystem>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <extras.hpp>
#include <init.hpp>

#ifdef __APPLE__
    #include <mach-o/dyld.h>
    #include <OpenCL/opencl.h>
#elif _WIN32
    #include "windows.h"
    #include "./CL/opencl.h"
#else 
    #include "windows.h"
    #include "./CL/opencl.h"
#endif

namespace Piro{
    namespace file_utilities{
        int writefile(std::string file_path, std::string line_to_write){
            // Create an output file stream (ofstream) object
            // std::ofstream outfile(file_path, std::ios::app);  // Open the file in append mode
            std::ofstream outfile(file_path);
            // Check if the file is open
            if (outfile.is_open()) {
                // Write the line to the file
                outfile << line_to_write << std::endl;

                // Close the file stream
                outfile.close();
                // std::cout << "Line written to file successfully" << std::endl;
            } else {
                Piro::Logger::info("Failed to open the file");
                // std::cerr << "Failed to open the file." << std::endl;
            }
            return 0;
        }

        int create_directory(std::string directoryname){
            std::filesystem::path dir(directoryname);
            Piro::Logger::info("Creating directory");
            try {
                if (std::filesystem::exists(dir)) {
                    Piro::Logger::info("Directory already exists: ", dir);
                } else {
                    if (std::filesystem::create_directory(dir)) {
                        Piro::Logger::info("Directory created successfully: ", dir);
                    } else {
                        Piro::Logger::info("Failed to create directory: ", dir);
                    }
                }
            } catch (const std::filesystem::filesystem_error& e) {
                Piro::Logger::info("Error: ", e.what());
            }
            return 0;
        }

        int delete_directory(std::string folderPath){
            try {
                if (std::filesystem::exists(folderPath)) {
                    std::filesystem::remove_all(folderPath);
                    Piro::Logger::info("Folder deleted successfully.");
                } else {
                    Piro::Logger::info("Folder does not exist.");
                }
            } catch (const std::filesystem::filesystem_error& e) {
                Piro::Logger::info("Error: ", e.what());
            }

            return 0;
        }

        int get_exec_directory(){
            
            #ifdef __APPLE__
                char path[1024];
                uint32_t size = sizeof(path);
                if (_NSGetExecutablePath(path, &size) == 0) {
                    std::filesystem::path exe_path = path;
                    current_path = exe_path.parent_path();
                    //std::cout << "Executable directory: " current_path.string() << std::endl;
                } else {
                    std::cerr << "Buffer size is too small; need size " << size << std::endl;
                }
            #else
                char path[1024];
                // Get the executable path
                DWORD size = GetModuleFileName(NULL, path, 1024);
                if (size > 0 && size < MAX_PATH) {
                std::filesystem::path exe_path(path);

                std::string exe_path_string = exe_path.string();
                std::replace(exe_path_string.begin(), exe_path_string.end(), '\\', '/');

                // Convert the modified string back to a filesystem path
                std::filesystem::path modified_exe_path = exe_path_string;
                current_path = modified_exe_path.parent_path();

                std::cout << "Original executable path: " << exe_path.string() << std::endl;
                std::cout << "Executable directory : " << current_path.string() << std::endl;
            } else {
                std::cerr << "Failed to retrieve executable path. Error code: " << GetLastError() << std::endl;
            }

            #endif
            return 0;
        }
    }  
}