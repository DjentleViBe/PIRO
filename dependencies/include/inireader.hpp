#ifndef INIREADER_HPP
#define INIREADER_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

// Trim whitespace from the beginning and end of a string
inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

class IniReader {
public:
    IniReader(const std::string& filename) {
        parseINI(filename);
    }

    std::string get(const std::string& section, const std::string& key, const std::string& defaultValue = "") const {
        auto sectIt = iniData.find(section);
        if (sectIt != iniData.end()) {
            auto keyIt = sectIt->second.find(key);
            if (keyIt != sectIt->second.end()) {
                return keyIt->second;
            }
        }
        return defaultValue;
    }
    
    void print() const {
        for (const auto& section : iniData) {
            std::cout << "[" << section.first << "]" << std::endl;
            for (const auto& kv : section.second) {
                std::cout << kv.first << "=" << kv.second << std::endl;
            }
        }
    }

private:
    std::map<std::string, std::map<std::string, std::string>> iniData;

    void parseINI(const std::string& filename) {
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
};

#endif // INIREADER_HPP