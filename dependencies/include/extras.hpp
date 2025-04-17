#ifndef extras_hpp
#define extras_hpp

#include <iostream>
#include <string>
#include <vector>
#include "datatypes.hpp"
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
void print_time();
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
std::vector<U> copyCL_offset(cl_command_queue queue, cl_mem memC, std::vector<U> Lap, int offset, int N, cl_event *event6) {
    if(Lap.size() < offset + N){
        // std::cout << "inserting extra : " <<  Lap.size() << ", N = "<< N << std::endl;
        Lap.insert(Lap.end(), offset + N - Lap.size(), 0.0); 
        
    }
    size_t offset_size = sizeof(U) * offset;     
    clEnqueueReadBuffer(queue, memC, CL_TRUE, offset_size,
                        sizeof(U) * N, Lap.data() + offset, 0, NULL, event6);
    
    clWaitForEvents(1, event6);
    return Lap;
}
void printCL(cl_mem memC, int N, int type);
void printCLArray(cl_mem memC, int N, int type);
void csr_to_dense_and_print(const std::vector<int>& row_pointer,
    const std::vector<int>& columns,
    const std::vector<float>& values,
    int N);
uint64_t nextPowerOf2(uint64_t N);
#endif