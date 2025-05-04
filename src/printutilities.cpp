#include <extras.hpp>
#include <iomanip>
#include <methods.hpp>

namespace Piro::PrintUtilities{
    void printMatrix(const std::vector<std::vector<float>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                Piro::Logger::debug_print(matrix[i][j], " ");
            }
            Piro::Logger::debug_print("\n");
        }
        Piro::Logger::debug_print("\n");
    }

    void printMatrix(const std::vector<std::vector<int>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                Piro::Logger::debug_print(matrix[i][j], " ");
            }
            Piro::Logger::debug_print("\n");
        }
        Piro::Logger::debug_print("\n");
    }

    void printVector(const std::vector<float>& vec){
        for (int i = 0; i < vec.size(); ++i) {
            Piro::Logger::debug_print(vec[i], " ");
        }
        Piro::Logger::debug_print("\n");
    }

    void printVector(const std::vector<int>& vec){
        for (int i = 0; i < vec.size(); ++i) {
            Piro::Logger::debug_print(vec[i], " ");
        }
        Piro::Logger::debug_print("\n");
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
            Piro::Logger::debug_print(array[i], " ");
        }
    }

    void printCL(cl_mem memC, int N, int type){
        if (type == 0){
            std::vector<int> hostValues(N);
            clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                                        sizeof(int) * N, hostValues.data(), 0, NULL, NULL);
            for (size_t i = 0; i < hostValues.size(); ++i) {
                Piro::Logger::debug_print(hostValues[i], " ");
                }
        }
        else if(type ==1){
            std::vector<float> hostValues(N);
            clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                                        sizeof(float) * N, hostValues.data(), 0, NULL, NULL);
            for (size_t i = 0; i < hostValues.size(); ++i) {
                Piro::Logger::debug_print(round(hostValues[i] * 10) / 10.0, " ");
                }
        }
        Piro::Logger::debug_print("\n");
    }

    void printCLArray(cl_mem memC, int N, int type){
        if (type == 0){
            std::vector<int> hostValues(N * N);
            clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                                        sizeof(int) * N * N, hostValues.data(), 0, NULL, NULL);
            for (size_t i = 0; i < hostValues.size(); ++i) {
                Piro::Logger::debug_print(hostValues[i], " ");
                if(i % (N - 1) == 0){
                    Piro::Logger::debug_print("\n");
                }
            }
        }
        else if(type ==1){
            std::vector<float> hostValues(N * N);
            clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                                        sizeof(float) * N * N, hostValues.data(), 0, NULL, NULL);
            for (size_t i = 0; i < hostValues.size(); ++i) {
                    
                Piro::Logger::debug_print(std::round(hostValues[i] * 10000.0) / 10000.0, "\t\t");
                if((i + 1) % (N) == 0){
                    Piro::Logger::debug_print("\n");
                }
            }
        }
        Piro::Logger::debug_print("\n");
    }

    void hash_to_dense_and_print(std::vector<int> Hashkeys, std::vector<float> HashValues, int N, int TABLE_SIZE){
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                float val = Piro::Methods::lookup(i, j, N, Hashkeys, HashValues, TABLE_SIZE);
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
        Piro::Logger::debug_print("\n");
        }
    }
}