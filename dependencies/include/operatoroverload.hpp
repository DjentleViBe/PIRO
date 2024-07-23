// operator_overload.hpp

#ifndef OPERATOROVERLOADHPP
#define OPERATOROVERLOADHPP
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#elif _WIN32
    #include "../dependencies/include/CL/opencl.h"
#else
    #include "../dependencies/include/CL/opencl.h"
#endif

#include <vector>

std::vector<float> operator+(const std::vector<float> &v1, const std::vector<float> &v2);
std::vector<std::vector<float>> operator+(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2);
std::vector<float> operator-(const std::vector<float> &v1, const std::vector<float> &v2);
std::vector<std::vector<float>> operator-(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2);
//std::vector<float> operator*(const std::vector<float> &v1, const std::vector<float> &v2);





cl_mem operator*(cl_mem partA, cl_mem partB);
//std::vector<std::vector<float>> operator*(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2);
//std::vector<float> operator/(const std::vector<float> &v1, const std::vector<float> &v2);
//std::vector<std::vector<float>> operator/(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2);
#endif // OPERATOR_OVERLOAD_HPP