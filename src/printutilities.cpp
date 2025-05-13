#include <extras.hpp>
#include <iomanip>
#include <methods.hpp>
#include <preprocess.hpp>
#include <logger.hpp>
#include <datatypes.hpp>
#include <cmath>
#include <chrono>
#include <CL/opencl.h>
#include <openclutilities.hpp>
#include <iostream>

void Piro::print_utilities::printMatrix(const std::vector<std::vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Piro::logger::debug_print(matrix[i][j], " ");
        }
        Piro::logger::debug_print("\n");
    }
    Piro::logger::debug_print("\n");
}

void Piro::print_utilities::printMatrix(const std::vector<std::vector<int>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Piro::logger::debug_print(matrix[i][j], " ");
        }
        Piro::logger::debug_print("\n");
    }
    Piro::logger::debug_print("\n");
}

void Piro::print_utilities::printVector(const std::vector<float>& vec){
    for (int i = 0; i < vec.size(); ++i) {
        Piro::logger::debug_print(vec[i], " ");
    }
    Piro::logger::debug_print("\n");
}

void Piro::print_utilities::printVector(const std::vector<int>& vec){
    for (int i = 0; i < vec.size(); ++i) {
        Piro::logger::debug_print(vec[i], " ");
    }
    Piro::logger::debug_print("\n");
}

void Piro::print_utilities::print_time() {
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

void Piro::print_utilities::printArray(const float *matrix, int rows, int cols){
    std::cout << "Printing array" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.1f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    std::cout << "\n";
}

void Piro::print_utilities::printCL(cl_mem memC, int N, int type){
    Piro::kernels& kernels = Piro::kernels::getInstance();
    if (type == 0){
        std::vector<int> hostValues(N);
        clEnqueueReadBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), memC, CL_TRUE, 0,
                                    sizeof(int) * N, hostValues.data(), 0, NULL, NULL);
        for (size_t i = 0; i < hostValues.size(); ++i) {
            Piro::logger::debug_print(hostValues[i], " ");
            }
    }
    else if(type ==1){
        std::vector<float> hostValues(N);
        clEnqueueReadBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), memC, CL_TRUE, 0,
                                    sizeof(float) * N, hostValues.data(), 0, NULL, NULL);
        for (size_t i = 0; i < hostValues.size(); ++i) {
            Piro::logger::debug_print(round(hostValues[i] * 10) / 10.0, " ");
            }
    }
    Piro::logger::debug_print("\n");
}

void Piro::print_utilities::printCLArray(cl_mem memC, int N, int type){
    Piro::kernels& kernels = Piro::kernels::getInstance();
    if (type == 0){
        std::vector<int> hostValues(N * N);
        clEnqueueReadBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), memC, CL_TRUE, 0,
                                    sizeof(int) * N * N, hostValues.data(), 0, NULL, NULL);
        for (size_t i = 0; i < hostValues.size(); ++i) {
            Piro::logger::debug_print(hostValues[i], " ");
            if(i % (N - 1) == 0){
                Piro::logger::debug_print("\n");
            }
        }
    }
    else if(type ==1){
        std::vector<float> hostValues(N * N);
        clEnqueueReadBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), memC, CL_TRUE, 0,
                                    sizeof(float) * N * N, hostValues.data(), 0, NULL, NULL);
        for (size_t i = 0; i < hostValues.size(); ++i) {
                
            Piro::logger::debug_print(std::round(hostValues[i] * 10000.0) / 10000.0, "\t\t");
            if((i + 1) % (N) == 0){
                Piro::logger::debug_print("\n");
            }
        }
    }
    Piro::logger::debug_print("\n");
}

void Piro::print_utilities::hash_to_dense_and_print(std::vector<int> Hashkeys, std::vector<float> HashValues, int N, int TABLE_SIZE){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            float val = Piro::methods::lookup(i, j, N, Hashkeys, HashValues, TABLE_SIZE);
            Piro::logger::debug_print(std::round(val * 100.0f) / 100.0f, "\t");
            // std::cout << std::fixed << std::setprecision(2) << val << "\t";
        }
        Piro::logger::debug_print("\n");
    }
}

void Piro::print_utilities::csr_to_dense_and_print(const std::vector<int>& row_pointer,
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
    Piro::logger::debug_print("\n");
    }
}
