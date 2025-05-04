#ifndef fileutilities_hpp
#define fileutilities_hpp
#include <string>

namespace Piro::FileUtilities{
    int writefile(std::string file_path, std::string line_to_write);
    int create_directory(std::string directoryname);
    int delete_directory(std::string folderPath);
    int get_exec_directory();
}

#endif