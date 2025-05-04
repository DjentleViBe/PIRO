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
#include <iomanip> 

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
              << std::setfill('0') << std::setw(1) << microseconds / 100 << "\t";
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

void hash_to_dense_and_print(std::vector<int> Hashkeys, std::vector<float> HashValues, int N, int TABLE_SIZE){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            float val = lookup(i, j, N, Hashkeys, HashValues, TABLE_SIZE);
            Piro::Logger::debug_print(std::round(val * 100.0f) / 100.0f, "\t");
            // std::cout << std::fixed << std::setprecision(2) << val << "\t";
        }
        Piro::Logger::debug_print("\n");
    }
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
            std::cout << std::fixed << std::setprecision(2) << val << "\t";
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


