#include <CL/opencl.h>
#include <preprocess.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <methods.hpp>
#include <openclutilities.hpp>
#include <logger.hpp>
#include <cmath>
#include <operatoroverload.hpp>
#include <matrixoperations.hpp>

namespace Piro{
    CLBuffer CLBuffer::operator=(const std::vector<CLBuffer>& other){

        cl_int err;
        int N = MP.n[0] * MP.n[1] * MP.n[2];
        int P = 1;
        std::vector<float>A(N, 0.0);
        CLBuffer partC, partD;
        partC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);
        partD.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * N, A.data(), &err);
        size_t globalWorkSize[1] = { (size_t)N };
        if(SP.timescheme == 11){
            // Forward Euler
            std::cout << "Forward Euler" << std::endl;

            err |= clSetKernelArg(kernellaplaciansparseMatrixMultiplyCSR, 0, sizeof(cl_int), &N);
            err |= clSetKernelArg(kernellaplaciansparseMatrixMultiplyCSR, 1, sizeof(cl_int), &N);
            err |= clSetKernelArg(kernellaplaciansparseMatrixMultiplyCSR, 2, sizeof(cl_int), &P);
            err |= clSetKernelArg(kernellaplaciansparseMatrixMultiplyCSR, 3, sizeof(cl_mem), &other[2].buffer);
            err |= clSetKernelArg(kernellaplaciansparseMatrixMultiplyCSR, 4, sizeof(cl_mem), &other[1].buffer);
            err |= clSetKernelArg(kernellaplaciansparseMatrixMultiplyCSR, 5, sizeof(cl_mem), &other[0].buffer);
            err |= clSetKernelArg(kernellaplaciansparseMatrixMultiplyCSR, 6, sizeof(cl_float), &SP.timestep);
            err |= clSetKernelArg(kernellaplaciansparseMatrixMultiplyCSR, 7, sizeof(cl_mem), &this->buffer);
            err |= clSetKernelArg(kernellaplaciansparseMatrixMultiplyCSR, 8, sizeof(cl_mem), &partC.buffer);
            err |= clSetKernelArg(kernellaplaciansparseMatrixMultiplyCSR, 9, sizeof(cl_int), &MP.n[0]);
            err |= clSetKernelArg(kernellaplaciansparseMatrixMultiplyCSR, 10, sizeof(cl_int), &MP.n[1]);
            err |= clSetKernelArg(kernellaplaciansparseMatrixMultiplyCSR, 11, sizeof(cl_int), &N);
            
            err = clEnqueueNDRangeKernel(queue, kernellaplaciansparseMatrixMultiplyCSR, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
            clFinish(queue);
            err |= clSetKernelArg(kernel_math[0], 0, sizeof(cl_mem), &partD.buffer);
            err |= clSetKernelArg(kernel_math[0], 2, sizeof(cl_mem), &partC.buffer);
            err |= clSetKernelArg(kernel_math[0], 1, sizeof(cl_mem), &this->buffer);
            err |= clSetKernelArg(kernel_math[0], 3, sizeof(cl_uint), &N);
            err = clEnqueueNDRangeKernel(queue, kernel_math[0], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        }

        else if(SP.timescheme == 12){
            std::cout << "Backward Euler" << std::endl;
                if(SP.solverscheme == 27){
                    Piro::Logger::info("LU Decomposition");
                    if(RHS_INIT == false){
                        Piro::matrix_operations::lu_decomposition_HTLF(other);
                        RHS_INIT = true;
                    }
                    Piro::Logger::debug("RHS_INIT end" );
                }
        }
        return partD;
    }

    CLBuffer operator*(CLBuffer partA, CLBuffer partB){
        cl_int err;
        int N = MP.n[0] * MP.n[1] * MP.n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernel_math[2], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernel_math[2], 1, sizeof(cl_mem), &partA.buffer);
        err |= clSetKernelArg(kernel_math[2], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernel_math[2], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(queue, kernel_math[2], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        
        return partC;
    }

    CLBuffer operator*(const float partA, CLBuffer partB){
        cl_int err;
        int N = MP.n[0] * MP.n[1] * MP.n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernel_math[6], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernel_math[6], 1, sizeof(cl_mem), &partA);
        err |= clSetKernelArg(kernel_math[6], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernel_math[6], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(queue, kernel_math[6], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator+(CLBuffer partA, CLBuffer partB){
        cl_int err;
        int N = MP.n[0] * MP.n[1] * MP.n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernel_math[0], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernel_math[0], 1, sizeof(cl_mem), &partA.buffer);
        err |= clSetKernelArg(kernel_math[0], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernel_math[0], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(queue, kernel_math[0], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator+(const float partA, CLBuffer partB){
        cl_int err;
        int N = MP.n[0] * MP.n[1] * MP.n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernel_math[4], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernel_math[4], 1, sizeof(cl_mem), &partA);
        err |= clSetKernelArg(kernel_math[4], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernel_math[4], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(queue, kernel_math[4], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }
        
    CLBuffer operator-(CLBuffer partA, CLBuffer partB){
        cl_int err;
        int N = MP.n[0] * MP.n[1] * MP.n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernel_math[1], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernel_math[1], 1, sizeof(cl_mem), &partA.buffer);
        err |= clSetKernelArg(kernel_math[1], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernel_math[1], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(queue, kernel_math[1], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator-(const float partA, CLBuffer partB){
        cl_int err;
        int N = MP.n[0] * MP.n[1] * MP.n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernel_math[5], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernel_math[5], 1, sizeof(cl_mem), &partA);
        err |= clSetKernelArg(kernel_math[5], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernel_math[5], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(queue, kernel_math[5], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator/(CLBuffer partA, CLBuffer partB){
        cl_int err;
        int N = MP.n[0] * MP.n[1] * MP.n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernel_math[3], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernel_math[3], 1, sizeof(cl_mem), &partA.buffer);
        err |= clSetKernelArg(kernel_math[3], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernel_math[3], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(queue, kernel_math[3], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator/(const float partA, CLBuffer partB){
        cl_int err;
        int N = MP.n[0] * MP.n[1] * MP.n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernel_math[7], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernel_math[7], 1, sizeof(cl_mem), &partA);
        err |= clSetKernelArg(kernel_math[7], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernel_math[7], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(queue, kernel_math[7], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }
};
