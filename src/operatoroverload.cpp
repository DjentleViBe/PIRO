#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#elif _WIN32
    #include "../dependencies/include/CL/opencl.h"
#else
    #include "../dependencies/include/CL/opencl.h"
#endif
#include <vector>
#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/operatoroverload.hpp"
// Overloading the + operator for std::vector<float>
std::vector<float> operator+(const std::vector<float> &v1, const std::vector<float> &v2) {
    Giro::MathOperations add;
    return add.addVectors(v1, v2);
}

std::vector<std::vector<float>> operator+(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2) {
    Giro::MathOperations addM;
    return addM.addMatrices(v1, v2);
}

std::vector<float> operator-(const std::vector<float> &v1, const std::vector<float> &v2) {
    Giro::MathOperations sub;
    return sub.subtractVectors(v1, v2);
}

std::vector<std::vector<float>> operator-(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2) {
    Giro::MathOperations subM;
    return subM.subtractMatrices(v1, v2);
}

/*
cl_mem operator*(cl_mem partA, cl_mem partB) {
    cl_int err;
    int N = MP.n[0] * MP.n[1] * MP.n[2];
    size_t globalWorkSize[1] = { (size_t)N };
    err |= clSetKernelArg(kernel_multiplyVec, 0, sizeof(cl_mem), &partA);
    err |= clSetKernelArg(kernel_multiplyVec, 1, sizeof(cl_mem), &partB);
    err = clEnqueueNDRangeKernel(queue, kernel_multiplyVec, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    
    return partA;
}*/

/*
std::vector<std::vector<float>> operator*(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2) {
    Giro::MathOperations mulM;
    return mulM.multiplyMatrices(v1, v2);
}*/

/*
std::vector<float> operator/(const std::vector<float> &v1, const std::vector<float> &v2) {
    Giro::MathOperations div;
    return div.divideVectors(v1, v2);
}

std::vector<std::vector<float>> operator/(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2) {
    Giro::MathOperations divM;
    return divM.divideMatrices(v1, v2);
}*/
