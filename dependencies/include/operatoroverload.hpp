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
#include "preprocess.hpp"
#include <vector>

std::vector<float> operator+(const std::vector<float> &v1, const std::vector<float> &v2);
std::vector<std::vector<float>> operator+(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2);
std::vector<float> operator-(const std::vector<float> &v1, const std::vector<float> &v2);
std::vector<std::vector<float>> operator-(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2);

class CLBuffer{
    public:
        cl_mem buffer;
        friend CLBuffer operator*(CLBuffer partA, CLBuffer partB){
            cl_int err;
            int N = MP.n[0] * MP.n[1] * MP.n[2];
            std::vector<float>A(N, 0.0);
            CLBuffer partC;
            partC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N, A.data(), &err);

            
            size_t globalWorkSize[1] = { (size_t)N };
            err |= clSetKernelArg(kernel_multiplyVec, 0, sizeof(cl_mem), &partC.buffer);
            err |= clSetKernelArg(kernel_multiplyVec, 1, sizeof(cl_mem), &partA.buffer);
            err |= clSetKernelArg(kernel_multiplyVec, 2, sizeof(cl_mem), &partB.buffer);
            err |= clSetKernelArg(kernel_multiplyVec, 3, sizeof(cl_uint), &N);
            err = clEnqueueNDRangeKernel(queue, kernel_multiplyVec, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

            return partC;
        }

        friend CLBuffer operator+(CLBuffer partA, CLBuffer partB){
            cl_int err;
            int N = MP.n[0] * MP.n[1] * MP.n[2];
            std::vector<float>A(N, 0.0);
            CLBuffer partC;
            partC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N, A.data(), &err);

            
            size_t globalWorkSize[1] = { (size_t)N };
            err |= clSetKernelArg(kernel_addVec, 0, sizeof(cl_mem), &partC.buffer);
            err |= clSetKernelArg(kernel_addVec, 1, sizeof(cl_mem), &partA.buffer);
            err |= clSetKernelArg(kernel_addVec, 2, sizeof(cl_mem), &partB.buffer);
            err |= clSetKernelArg(kernel_addVec, 3, sizeof(cl_uint), &N);
            err = clEnqueueNDRangeKernel(queue, kernel_addVec, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

            return partC;
        }
};

namespace Giro{
    class CellDataGPU{
        public:
            std::vector<CLBuffer> values_gpu;
    };
};

// cl_mem operator*(cl_mem partA, cl_mem partB);
//std::vector<float> operator*(const std::vector<float> &v1, const std::vector<float> &v2);
//std::vector<std::vector<float>> operator*(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2);
//std::vector<float> operator/(const std::vector<float> &v1, const std::vector<float> &v2);
//std::vector<std::vector<float>> operator/(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2);
#endif // OPERATOR_OVERLOAD_HPP