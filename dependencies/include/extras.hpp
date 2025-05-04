#ifndef extras_hpp
#define extras_hpp

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <datatypes.hpp>
#include <preprocess.hpp>
#include <printutilities.hpp>
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

namespace Piro{
    class Logger {
        public:
            template<typename... Args>
            static void info(const Args&... args) {
                if(debuginfo >= 0){
                    Piro::PrintUtilities::print_time();
                    ((std::cout << args << " "), ...);
                    std::cout << "" << std::endl;
                }
            }
            
            template<typename... Args>
            static void error(const Args&... args) {
                if(debuginfo >= 3){
                    Piro::PrintUtilities::print_time();
                    ((std::cout << args << " "), ...);
                    std::cout << "" << std::endl;
                }
            }

            template<typename... Args>
            static void debug(const Args&... args) {
                if(debuginfo >= 1){
                    Piro::PrintUtilities::print_time();
                    ((std::cout << args << " "), ...);
                    std::cout << "" << std::endl;
                }
            }
            template<typename... Args>
            static void debug_print(const Args&... args) {
                if(debuginfo >= 1){
                    ((std::cout << args << " "), ...);
                    // std::cout << "" << std::endl;
                }
            }

            // Basic case with a single argument
            template<typename T>
            static void info_debug(const T& arg) {
                std::cout << arg;
            }
            
            // Specialization for vectors
            template<typename T>
            static void info_debug(const std::vector<T>& vec) {
                for (const auto& item : vec) {
                    std::cout << item << " ";
                }
            }

            // Variadic template version for multiple arguments
            template<typename T, typename... Args>
            static void warning(const T& first, const Args&... args) {
                if(debuginfo >= 2){
                    Piro::PrintUtilities::print_time();
                    info_debug(first);  // Handle the first argument
                    info_debug(args...); // Recursive call for remaining arguments
                    std::cout << "" << std::endl;
                }
                
            }
        };
}

std::vector<int> flattenvector(std::vector<std::vector<int>> twoDVector);

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
uint64_t nextPowerOf2(uint64_t N);

inline bool is_prime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}

inline int next_prime(int n) {
    while (!is_prime(n)) ++n;
    return n;
}

#endif