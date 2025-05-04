#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <extras.hpp>
#include <datatypes.hpp>
#include <init.hpp>
#include <operatoroverload.hpp>
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

std::vector<int> flattenvector(std::vector<std::vector<int>> twoDVector){
    std::vector<int> flatVector;
    for (const auto& row : twoDVector) {
        for (int elem : row) {
            flatVector.push_back(elem);
        }
    }
    return flatVector;
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


