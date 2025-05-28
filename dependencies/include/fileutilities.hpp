#ifndef fileutilities_hpp
#define fileutilities_hpp
#include <string>
#include <map>
#include <filesystem>
#include <vector>

namespace Piro::file_utilities{
    /**
     * @brief Write file.
     *
     * This function writes content to a file.
     *
     * @param file_path Path of the file along with the file name.
     * @param line_to_write Content to write.
     * @return 0 upon SUCCESS.
     */
    int writefile(std::string file_path, std::string line_to_write);
    /**
     * @brief Create a directory.
     *
     * This function create a directory if it does not exist.
     *
     * @param directoryname Name of the directory appended to the file path.
     * @return 0 upon SUCCESS.
     */
    int create_directory(std::string directoryname);
    /**
     * @brief Deletes a directory.
     *
     * This function deletes a directory if it exists.
     *
     * @param folderPath name of the directory appended to the file path.
     * @return 0 upon SUCCESS
     */
    int delete_directory(std::string folderPath);
        /**
     * @brief Fetche executable directory.
     *
     * This function fetches the location of the executable file which is 
     * launched and assigns it to VAR = current_path. The function is
     * adpated to work with MACOS, WINDOWS and LINUX OS.
     * 
     * @return 0 upon SUCCESS
     */
    int get_exec_directory();
    inline std::filesystem::path current_path;
    /**
     * @brief File reader class.
     *
     * This file class for reading content from given file.
     */
    class IniReader {
        public:
            IniReader(const std::string& filename);
            std::string trim(const std::string& str);
            std::string get(const std::string& section, const std::string& key, const std::string& defaultValue = "") const;
            void print() const;
            static std::vector<std::vector<int>> parsemesh(const std::string& filename);
        private:
            std::map<std::string, std::map<std::string, std::string>> iniData;
            void parseINI(const std::string& filename);
            
    };
}

#endif