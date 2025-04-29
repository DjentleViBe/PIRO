#ifndef extras_hpp
#define extras_hpp

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
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
        static void error(const Args&... args) {
            if(debuginfo >= 3){
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
                print_time();
                info_debug(first);  // Handle the first argument
                info_debug(args...); // Recursive call for remaining arguments
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
    int attempts = 0;
    int first_deleted = -1;
    // Linear probing to find the key
    while (Hash_keys_V[hash_index] != -1 && Hash_keys_V[hash_index] != index) {
        if (Hash_keys_V[hash_index] == -2 && first_deleted == -1) {
            first_deleted = hash_index;  // Remember the first deleted slot
        } 
        hash_index = (hash_index + 1) % TABLE_SIZE;
        attempts++;
        if (attempts >= TABLE_SIZE) {
            // Logger::info("Error - lookhash [", hash_index, "]: Hash table [", TABLE_SIZE, "] is full. \n\t\t\t\t\t\t\tAttempts =", attempts,". \n\t\t\t\t\t\t\t");
            // Table is full, handle appropriately
            // std::exit(1);
            return 0.0; // Return error code
        }
    }

    if (Hash_keys_V[hash_index] == index) {
        // Key already exists → update value
        return Hash_val_V[hash_index];
    }
    return 0.0;
    // Key not found
}

inline float query(int index, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V, int TABLE_SIZE) {
    int hash_index = index % TABLE_SIZE;
    int attempts = 0;
    int first_deleted = -1;
    // Linear probing to find the key
    while (Hash_keys_V[hash_index] != -1 && Hash_keys_V[hash_index] != index) {
        if (Hash_keys_V[hash_index] == -2 && first_deleted == -1) {
            first_deleted = hash_index;  // Remember the first deleted slot
        } 
        
        hash_index = (hash_index + 1) % TABLE_SIZE;
        attempts++;
        if (attempts >= TABLE_SIZE) {
            // Logger::info("Error - lookhash [", hash_index, "]: Hash table [", TABLE_SIZE, "] is full. \n\t\t\t\t\t\t\tAttempts =", attempts,". \n\t\t\t\t\t\t\t");
            // Table is full, handle appropriately
            // std::exit(1);
            return 0.0; // Return error code
        }
    }
    if (Hash_keys_V[hash_index] == index) {
        std::cout << "Index :" << Hash_keys_V[hash_index] <<  ", hash: "<<hash_index<<", Value = "<< Hash_val_V[hash_index] << std::endl; // key found
        return 0.0;
    }
    else if(Hash_keys_V[hash_index] == -2){
        int insert_index = (first_deleted != -1) ? first_deleted : hash_index;
        std::cout << "Index :" << Hash_keys_V[insert_index] <<  ", hash: "<< insert_index<<", Value = "<< Hash_val_V[hash_index] << std::endl; // key found
        
    }
    else{
        std::cout << "key not found" << std::endl;
    }
    
    // Key not found
    return 0.0; // or some sentinel value for "not found"
}

inline int sethash(int index, float val, int TABLE_SIZE, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V) {
    int hash_index = index % TABLE_SIZE;
    int attempts = 0;
    int first_deleted = -1;

    while (attempts < TABLE_SIZE) {
        if (Hash_keys_V[hash_index] == index) {
            // Key found → update
            Hash_val_V[hash_index] = val;
            return 0;
        }
        else if (Hash_keys_V[hash_index] == -1) {
            // Empty slot → insert
            int insert_index = (first_deleted != -1) ? first_deleted : hash_index;
            Hash_keys_V[insert_index] = index;
            Hash_val_V[insert_index] = val;
            return 0;
        }
        else if (Hash_keys_V[hash_index] == -2 && first_deleted == -1) {
            // First deleted slot
            first_deleted = hash_index;
        }

        // Continue probing
        hash_index = (hash_index + 1) % TABLE_SIZE;
        attempts++;
    }

    // If full loop completed, insert at first_deleted if available
    if (first_deleted != -1) {
        Hash_keys_V[first_deleted] = index;
        Hash_val_V[first_deleted] = val;
        return 0;
    }

    // Table is truly full
    Logger::info("Error - sethash [", hash_index, "/", index, "]: Hash table [", TABLE_SIZE, "] is full. \n\t\t\t\t\t\t\tAttempts =", attempts,", first_deleted = ", first_deleted,". Try reducing LoadFactor inside hashtable.ini. \n\t\t\t\t\t\t\tAborting program");
    Logger::warning("hashkeys : ", Hash_keys_V);
    Logger::warning("hashvalues : ", Hash_val_V);
    hash_to_dense_and_print(Hash_keys_V, Hash_val_V, MP.n[0]*MP.n[1]*MP.n[2], TABLE_SIZE);
    std::exit(1);
    return -1;
}

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