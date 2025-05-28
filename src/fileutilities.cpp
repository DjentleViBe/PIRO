#include <filesystem>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <extras.hpp>
#include <init.hpp>
#include <logger.hpp>
#include <CL/opencl.h>
#include <fileutilities.hpp>
#include <map>

#ifdef __APPLE__
    #include <mach-o/dyld.h>
#elif _WIN32 || __CYGWIN__
    #include "windows.h"
#else
    #include <unistd.h>
    #include <limits.h>
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
                Piro::logger::info("Failed to open the file");
                // std::cerr << "Failed to open the file." << std::endl;
            }
            return 0;
        }

        int create_directory(std::string directoryname){
            std::filesystem::path dir(directoryname);
            Piro::logger::info("Creating directory");
            try {
                if (std::filesystem::exists(dir)) {
                    Piro::logger::info("Directory already exists: ", dir);
                } else {
                    if (std::filesystem::create_directory(dir)) {
                        Piro::logger::info("Directory created successfully: ", dir);
                    } else {
                        Piro::logger::info("Failed to create directory: ", dir);
                    }
                }
            } catch (const std::filesystem::filesystem_error& e) {
                Piro::logger::info("Error: ", e.what());
            }
            return 0;
        }

        int delete_directory(std::string folderPath){
            try {
                if (std::filesystem::exists(folderPath)) {
                    std::filesystem::remove_all(folderPath);
                    Piro::logger::info("Folder deleted successfully.");
                } else {
                    Piro::logger::info("Folder does not exist.");
                }
            } catch (const std::filesystem::filesystem_error& e) {
                Piro::logger::info("Error: ", e.what());
            }

            return 0;
        }

        int get_exec_directory(){
            
            #ifdef __APPLE__
                char path[1024];
                uint32_t size = sizeof(path);
                if (_NSGetExecutablePath(path, &size) == 0) {
                    std::filesystem::path exe_path = path;
                    Piro::file_utilities::current_path = exe_path.parent_path();
                    //std::cout << "Executable directory: " Piro::file_utilities::current_path.string() << std::endl;
                } else {
                    std::cerr << "Buffer size is too small; need size " << size << std::endl;
                }
            #elif _WIN32 || __CYGWIN__
                char path[1024];
                // Get the executable path
                DWORD size = GetModuleFileName(NULL, path, 1024);
                if (size > 0 && size < MAX_PATH) {
                    std::filesystem::path exe_path(path);

                    std::string exe_path_string = exe_path.string();
                    std::replace(exe_path_string.begin(), exe_path_string.end(), '\\', '/');

                    // Convert the modified string back to a filesystem path
                    std::filesystem::path modified_exe_path = exe_path_string;
                    Piro::file_utilities::current_path = modified_exe_path.parent_path();

                    std::cout << "Original executable path: " << exe_path.string() << std::endl;
                    std::cout << "Executable directory : " << Piro::file_utilities::current_path.string() << std::endl;
                } 
                else {
                    std::cerr << "Failed to retrieve executable path. Error code: " << GetLastError() << std::endl;
                }
            #else
                char path[PATH_MAX];
                ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
                if (count != -1) {
                    std::filesystem::path exe_path(std::string(path, count));
                    Piro::file_utilities::current_path = exe_path.parent_path();
                } 
                else {
                    std::cerr << "Failed to retrieve executable path using /proc/self/exe" << std::endl;
                }

            #endif
            return 0;
        }
        
        IniReader::IniReader(const std::string& filename) {
            parseINI(filename);
        }
        
        std::string IniReader::trim(const std::string& str) {
            size_t first = str.find_first_not_of(' ');
            if (first == std::string::npos) return "";
            size_t last = str.find_last_not_of(' ');
            return str.substr(first, (last - first + 1));
        }

        std::vector<std::vector<int>> IniReader::parsemesh(const std::string& filename){
            std::ifstream file(filename);
            std::vector<std::vector<int>> data;
            std::string line;

            while (std::getline(file, line)) {
                std::vector<int> row;
                std::stringstream ss(line);
                std::string value;

                while (std::getline(ss, value, ',')) {
                    // Remove leading/trailing spaces
                    value.erase(0, value.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);
                    row.push_back(std::stoi(value));
                }
                data.push_back(row);
            }
            return data;
        }

        void IniReader::parseINI(const std::string& filename) {
            std::ifstream file(filename);
            std::string line, section;

            if (!file.is_open()) {
                std::cerr << "Could not open the file!" << std::endl;
                return;
            }

            while (std::getline(file, line)) {
                line = trim(line);

                // Skip comments and empty lines
                if (line.empty() || line[0] == ';' || line[0] == '#') continue;

                // Section header
                if (line[0] == '[' && line[line.size() - 1] == ']') {
                    section = line.substr(1, line.size() - 2);
                } else {
                    // Key-value pair
                    size_t delimPos = line.find('=');
                    if (delimPos != std::string::npos) {
                        std::string key = trim(line.substr(0, delimPos));
                        std::string value = trim(line.substr(delimPos + 1));
                        iniData[section][key] = value;
                    }
                }
            }
            file.close();
        }
        
        std::string IniReader::get(const std::string& section, const std::string& key, const std::string& defaultValue) const {
            auto sectIt = iniData.find(section);
            if (sectIt != iniData.end()) {
                auto keyIt = sectIt->second.find(key);
                if (keyIt != sectIt->second.end()) {
                    return keyIt->second;
                }
            }
            return defaultValue;
        }
                
        void IniReader::print() const {
            for (const auto& section : iniData) {
                std::cout << "[" << section.first << "]" << std::endl;
                for (const auto& kv : section.second) {
                    std::cout << kv.first << "=" << kv.second << std::endl;
                }
            }
        }
        std::map<std::string, std::map<std::string, std::string>> iniData;
    }
}