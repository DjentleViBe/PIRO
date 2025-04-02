#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <iterator>
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/datatypes.hpp"
#include "../dependencies/include/init.hpp"
#include "operatoroverload.hpp"
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
#include <numeric>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <cstdint>

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
        std::cout << "Filed to open the file" << std::endl;
        // std::cerr << "Failed to open the file." << std::endl;
    }
    return 0;
}

int create_directory(std::string directoryname){
    std::filesystem::path dir(directoryname);
    std::cout << "Creating directory" << std::endl;
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

std::vector<uint> convertStringVectorToUInt(const std::vector<std::string>& stringVector) {
    std::vector<uint> intVector;
    
    for (const std::string& str : stringVector) {
        std::istringstream iss(str);
        uint num;
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
    for (int i = dex * floatVector.size() / 3; i < (dex + 1) * floatVector.size() / 3; ++i) {
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

int countWords(const std::string& str) {
    std::istringstream iss(str);
    return std::distance(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
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
        std::cout << round(vec[i] * 10000) / 10000.0 << " ";
    }
    std::cout << std::endl;
}

void printVector(const std::vector<int>& vec){
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

void print_time() {
    // Get current time as a time_point
    auto now = std::chrono::system_clock::now();

    // Convert to std::time_t to get the calendar time
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    // Get microseconds for 4-digit precision
    auto now_us = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    auto us = now_us.time_since_epoch().count() % 1000000; // Microseconds

    // Get milliseconds (the first three digits)
    int milliseconds = us / 1000;
    
    // Get microseconds (the last digit)
    int microseconds = us % 1000;

    // Convert to local time
    std::tm now_tm = *std::localtime(&now_time_t);

    // Print the time with 4-digit millisecond precision
    std::cout << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S") << '.'
              << std::setfill('0') << std::setw(3) << milliseconds
              << std::setfill('0') << std::setw(2) << microseconds / 10
              << std::setfill('0') << std::setw(1) << microseconds % 10 << "\t";
}

void printArray(float* array, uint size){
    std::cout << "Printing array" << std::endl;
    for (uint i = 0; i < size; i++) {
        std::cout << array[i] << " ";
    }
}

std::vector<int> flattenvector(std::vector<std::vector<int>> twoDVector){
    std::vector<int> flatVector;
    for (const auto& row : twoDVector) {
        for (int elem : row) {
            flatVector.push_back(elem);
        }
    }
    return flatVector;
}

void printCL(cl_mem memC, int N, int type){
    if (type == 0){
        std::vector<int> hostValues(N);
        clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                                    sizeof(int) * N, hostValues.data(), 0, NULL, NULL);
        for (size_t i = 0; i < hostValues.size(); ++i) {
            std::cout << hostValues[i] << " ";
            }
    }
    else if(type ==1){
        std::vector<float> hostValues(N);
        clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                                    sizeof(float) * N, hostValues.data(), 0, NULL, NULL);
        for (size_t i = 0; i < hostValues.size(); ++i) {
            std::cout << round(hostValues[i] * 10) / 10.0 << " ";
            }
    }
    std::cout << std::endl;
}

std::vector<float> copyCL(cl_mem memC, int N, int type){
    std::vector<float> hostValues(N);
    if (type == 0){
        clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                                    sizeof(int) * N, hostValues.data(), 0, NULL, NULL);

    }
    else if(type ==1){
        
        clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                                    sizeof(float) * N, hostValues.data(), 0, NULL, NULL);

    }
    clFinish(queue);
    return hostValues;
}

void printCLArray(cl_mem memC, int N, int type){
    if (type == 0){
        std::vector<int> hostValues(N * N);
        clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                                    sizeof(int) * N * N, hostValues.data(), 0, NULL, NULL);
        for (size_t i = 0; i < hostValues.size(); ++i) {
            std::cout << hostValues[i] << " ";
            if(i % (N - 1) == 0){
                std::cout << std::endl;
            }
            }
    }
    else if(type ==1){
        std::vector<float> hostValues(N * N);
        clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                                    sizeof(float) * N * N, hostValues.data(), 0, NULL, NULL);
        for (size_t i = 0; i < hostValues.size(); ++i) {
                
            std::cout << std::round(hostValues[i] * 10000.0) / 10000.0 << "\t\t";
            if((i + 1) % (N) == 0){
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

void csr_to_dense_and_print(const std::vector<int>& row_pointer,
    const std::vector<int>& columns,
    const std::vector<float>& values,
    int N) {
    // Initialize a dense matrix with zeros
    std::vector<std::vector<float>> dense_matrix(N, std::vector<float>(N, 0.0f));

    // Convert CSR to dense format
    for (int i = 0; i < N; i++) {
        for (int j = row_pointer[i]; j < row_pointer[i + 1]; j++) {
        dense_matrix[i][columns[j]] = values[j];
    }
}

    // Print the dense matrix
    for (const auto& row : dense_matrix) {
        for (float val : row) {
        std::cout << val << " ";
        }
    std::cout << "\n";
    }
}

uint64_t nextPowerOf2(uint64_t N) {
    if (N <= 1) return 1;
    N--;
    N |= N >> 1;
    N |= N >> 2;
    N |= N >> 4;
    N |= N >> 8;
    N |= N >> 16;
    N |= N >> 32;  // Safe for 64-bit numbers
    return N + 1;
}


