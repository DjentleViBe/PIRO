#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/init.hpp"
#include <mach-o/dyld.h>
#include <numeric>
#include <vector>

int writefile(std::string file_path, std::string line_to_write){
    // Create an output file stream (ofstream) object
    //std::ofstream outfile(file_path, std::ios::app);  // Open the file in append mode
    std::ofstream outfile(file_path);
    // Check if the file is open
    if (outfile.is_open()) {
        // Write the line to the file
        outfile << line_to_write << std::endl;

        // Close the file stream
        outfile.close();
        // std::cout << "Line written to file successfully" << std::endl;
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

int delete_directory(std::string folderPath){
    try {
        if (std::filesystem::exists(folderPath)) {
            std::filesystem::remove_all(folderPath);
            std::cout << "Folder deleted successfully." << std::endl;
        } else {
            std::cout << "Folder does not exist." << std::endl;
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

std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::size_t start = 0, end = 0;

    while ((end = str.find(delimiter, start)) != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + 1;
    }

    tokens.push_back(str.substr(start));
    return tokens;
}

std::vector<int> convertStringVectorToInt(const std::vector<std::string>& stringVector) {
    std::vector<int> intVector;
    
    for (const std::string& str : stringVector) {
        std::istringstream iss(str);
        int num;
        if (iss >> num) {
            intVector.push_back(num);
        } else {
            // Handle error if conversion fails
            std::cerr << "Error: Conversion failed for string '" << str << "'" << std::endl;
            // You may choose to throw an exception or handle the error in another way
        }
    }
    
    return intVector;
}

std::vector<float> convertStringVectorToFloat(const std::vector<std::string>& stringVector) {
    std::vector<float> intVector;
    
    for (const std::string& str : stringVector) {
        std::istringstream iss(str);
        float num;
        if (iss >> num) {
            intVector.push_back(num);
        } else {
            // Handle error if conversion fails
            std::cerr << "Error: Conversion failed for string '" << str << "'" << std::endl;
            // You may choose to throw an exception or handle the error in another way
        }
    }
    
    return intVector;
}

std::vector<std::string> floatScalarToString(const std::vector<float>& floatScalar) {
    std::vector<std::string> stringScalar;

    // Iterate through each float element and convert to string
    for (float value : floatScalar) {
        // Convert float to string
        std::ostringstream oss;
        oss << value;
        stringScalar.push_back(oss.str());
    }

    return stringScalar;
}

std::vector<std::string> floatVectorToString(const std::vector<float>& floatVector, int dex) {
    std::vector<std::string> stringVector;

    // Iterate through each float element and convert to string
    for (int i = dex * floatVector.size(); i <= (dex + 1) * floatVector.size(); ++i) {
        // Convert float to string
        std::ostringstream oss;
        oss << floatVector[i];
        stringVector.push_back(oss.str());
    }

    return stringVector;
}

std::string concatenateStrings(const std::vector<std::string>& strVector) {
    // Use std::accumulate to concatenate all strings in the vector
    return std::accumulate(strVector.begin(), strVector.end(), std::string(""));
}

std::string concatenateStrings2(const std::vector<std::string>& strVector) {
    // Use std::accumulate to concatenate all strings in the vector
 
    return std::accumulate(strVector.begin(), strVector.end(), std::string(""),
                           [](const std::string& acc, const std::string& str) {
                               return acc.empty() ? str : acc + " " + str;
                           });
}

int countSpaces(const std::string& str) {
    int spaceCount = 0;

    // Iterate through each character in the string
    for (char ch : str) {
        // Check if the character is a space
        if (ch == ' ') {
            spaceCount++;
        }
    }

    return spaceCount;
}

void printMatrix(const std::vector<std::vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printMatrix(const std::vector<std::vector<int>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printVector(const std::vector<float>& vec){
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

void printVector(const std::vector<int>& vec){
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}