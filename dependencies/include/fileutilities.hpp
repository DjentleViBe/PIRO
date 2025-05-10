#ifndef fileutilities_hpp
#define fileutilities_hpp
#include <string>
#include <map>

namespace Piro::file_utilities{
    int writefile(std::string file_path, std::string line_to_write);
    int create_directory(std::string directoryname);
    int delete_directory(std::string folderPath);
    int get_exec_directory();
    inline std::filesystem::path current_path;
    class IniReader {
        public:
            IniReader(const std::string& filename);
            std::string trim(const std::string& str);
            std::string get(const std::string& section, const std::string& key, const std::string& defaultValue = "") const;
            void print() const;
        private:
            std::map<std::string, std::map<std::string, std::string>> iniData;
            void parseINI(const std::string& filename);

    };
}

#endif