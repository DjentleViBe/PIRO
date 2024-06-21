#ifndef extras_hpp
#define extras_hpp

#include <string>

int writefile(std::string file_path, std::string line_to_write);
int create_directory(std::string directoryname);
int get_exec_directory();
#endif