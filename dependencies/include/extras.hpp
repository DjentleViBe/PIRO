#ifndef extras_hpp
#define extras_hpp

#include <iostream>
#include <string>
#include <vector>
#include "datatypes.hpp"
#include "preprocess.hpp"
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
extern int debuginfo;
void print_time();
class Logger {
    public:
        template<typename... Args>
        static void info(const Args&... args) {
            if(debuginfo >= 0){
                print_time();
                ((std::cout << args << " "), ...);
                std::cout << "" << std::endl;
            }
        }
        
        template<typename... Args>
        static void debug(const Args&... args) {
            if(debuginfo >= 1){
                print_time();
                ((std::cout << args << " "), ...);
                std::cout << "" << std::endl;
            }
        }
        
        template<typename... Args>
        static void error(const Args&... args) {
            if(debuginfo >= 3){
                print_time();
                ((std::cout << args << " "), ...);
                std::cout << "" << std::endl;
            }
        }

        template<typename... Args>
        static void warning(const Args&... args) {
            if(debuginfo >= 2){
                print_time();
                ((std::cout << args << " "), ...);
                std::cout << "" << std::endl;
            }
        }
    };
int writefile(std::string file_path, std::string line_to_write);
int create_directory(std::string directoryname);
int delete_directory(std::string folderPath);
int get_exec_directory();
std::vector<std::string> splitString(const std::string& str, char delimiter);
std::vector<int> convertStringVectorToInt(const std::vector<std::string>& stringVector);
std::vector<uint> convertStringVectorToUInt(const std::vector<std::string>& stringVector);
std::vector<float> convertStringVectorToFloat(const std::vector<std::string>& stringVector);
std::vector<std::string> floatVectorToString(const std::vector<float>& floatVector, int dex);
std::vector<std::string> floatScalarToString(const std::vector<float>& floatScalar);
std::string concatenateStrings(const std::vector<std::string>& strVector);
int countSpaces(const std::string& str);
std::string concatenateStrings2(const std::vector<std::string>& strVector);
void printMatrix(const std::vector<std::vector<float>>& matrix);
void printMatrix(const std::vector<std::vector<int>>& matrix);
void printVector(const std::vector<float>& vec);
void printVector(const std::vector<int>& vec);
void printArray(float* array, uint size);
std::vector<int> flattenvector(std::vector<std::vector<int>> twoDVector);
int countWords(const std::string& str);
template <typename T>
std::vector<T> copyCL(cl_command_queue queue, cl_mem memC, int N, cl_event *event6) {
    std::vector<T> hostValues(N);
    clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                        sizeof(T) * N, hostValues.data(), 0, NULL, event6);
    clFinish(queue);
    return hostValues;
}
template <typename U>
int copyCL_offset(cl_command_queue queue, cl_mem memC, std::vector<U>& Lap, int offset, int N, cl_event *event6) {
    size_t offset_size = sizeof(U) * offset;     
    clEnqueueReadBuffer(queue, memC, CL_FALSE, offset_size,
                        sizeof(U) * (N - offset), Lap.data() + offset, 0, NULL, event6);
    
    clWaitForEvents(1, event6);
    return 0;
}
void printCL(cl_mem memC, int N, int type);
void printCLArray(cl_mem memC, int N, int type);
void csr_to_dense_and_print(const std::vector<int>& row_pointer,
    const std::vector<int>& columns,
    const std::vector<float>& values,
    int N);
void hash_to_dense_and_print(std::vector<int> Hashkeys, std::vector<float> HashValues, int N, int TABLE_SIZE);
uint64_t nextPowerOf2(uint64_t N);
inline float lookup(int row, int col, int N, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V, int TABLE_SIZE) {
    int index = row * N + col;
    int hash_index = index % TABLE_SIZE;

    // Linear probing to find the key
    while (Hash_keys_V[hash_index] != -1) {
        if (Hash_keys_V[hash_index] == index) {
            return Hash_val_V[hash_index]; // key found
        }
        hash_index = (hash_index + 1) % TABLE_SIZE;
    }

    // Key not found
    return 0.0; // or some sentinel value for "not found"
}
inline int sethash(int index, float val, int TABLE_SIZE, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V){
    int hash_index = index % TABLE_SIZE;
    // Linear probing
    while (Hash_keys_V[hash_index] > -1 && Hash_keys_V[hash_index] != index) {
        hash_index = (hash_index + 1) % TABLE_SIZE;
    }

    Hash_keys_V[hash_index] = index;
    Hash_val_V[hash_index] = val;
    return 0;
}
#endif