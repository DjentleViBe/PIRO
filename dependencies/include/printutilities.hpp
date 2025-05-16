#ifndef printutilities_hpp
#define printutilities_hpp
#include <vector>
#include <CL/opencl.h>
#ifdef __APPLE__
    #include <mach-o/dyld.h>
#elif _WIN32
    #include "windows.h"
#endif
#include <datatypes.hpp>

namespace Piro::print_utilities{
    /**
     * @brief print matrix of float type.
     *
     * Prints a matrix of float type to the console.
     * @param matrix a 2D vector
     */
    void printMatrix(const std::vector<std::vector<float>>& matrix);
    /**
     * @brief print matrix of int type.
     *
     * Prints a matrix of int type to the console.
     * @param matrix a 2D vector
     */
    void printMatrix(const std::vector<std::vector<int>>& matrix);
    /**
     * @brief print vector of float type.
     *
     * Prints a vector of float typeto the console.
     * @param vec a 1D vector
     */
    void printVector(const std::vector<float>& vec);
    /**
     * @brief print vector of int type.
     *
     * Prints a vector of int type to the console.
     * @param vec a 1D vector
     */
    void printVector(const std::vector<int>& vec);
    /**
     * @brief print an array of float type.
     *
     * Prints an array to the console.
     * @param array a pointer to the array location.
     * @param row number of rows of the array.
     * @param cols number of columns of the array.
     */
    void printArray(const float *matrix, int rows, int cols);
    /**
     * @brief print array of int / float type.
     *
     * Prints an array to the console.
     * @param memC OpenCL memory location 
     * @param N size of the array
     * @param array type
     */
    void printCL(cl_mem memC, int N, int type);
    /**
     * @brief print array of int / float type.
     *
     * Prints an array to the console.
     * @param a OpenCL memory location 
     * @param N size of the array
     * @param array type
     */
    void printCLArray(cl_mem memC, int N, int type);
    /**
     * @brief print CSR matrix in dense format.
     *
     * Prints an matrix in CSR format as dense format to the console.
     * @param row_pointer rowpointer
     * @param columns column 
     * @param values values  
     */
    void csr_to_dense_and_print(const std::vector<int>& row_pointer,
                                const std::vector<int>& columns,
                                const std::vector<float>& values,
                                int N);
    /**
     * @brief print a dense matrix from a Hash Table.
     *
     * Prints an matrix in dense format.
     * @param Hashkeys hash keys
     * @param HashValues hash values 
     * @param N total number of rows
     * @param TABLE_SIZE hash table size  
     */
    void hash_to_dense_and_print(std::vector<int> Hashkeys, std::vector<float> HashValues, int N, int TABLE_SIZE);
    /**
     * @brief Prints the current system time to console.
     *
     * retrieves the current system time and prints it
     * in a human-readable format (e.g., HH:MM:SS:MS:ms)
     */
    void print_time();
}
#endif