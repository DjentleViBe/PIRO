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
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        int P = 1;
        std::vector<float>A(N, 0.0);
        CLBuffer partC, partD;
        partC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);
        partD.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * N, A.data(), &err);
        size_t globalWorkSize[1] = { (size_t)N };
        Piro::SolveParams& SP = Piro::SolveParams::getInstance();
        float timestep = SP.getvalue<float>(Piro::SolveParams::TIMESTEP);
        if(SP.getvalue<int>(Piro::SolveParams::TIMESCHEME) == 11){
            // Forward Euler
            // std::cout << "Forward Euler" << std::endl;

            err |= clSetKernelArg(kernels::kernel[8], 0, sizeof(cl_int), &N);
            err |= clSetKernelArg(kernels::kernel[8], 1, sizeof(cl_int), &N);
            err |= clSetKernelArg(kernels::kernel[8], 2, sizeof(cl_int), &P);
            err |= clSetKernelArg(kernels::kernel[8], 3, sizeof(cl_mem), &other[2].buffer);
            err |= clSetKernelArg(kernels::kernel[8], 4, sizeof(cl_mem), &other[1].buffer);
            err |= clSetKernelArg(kernels::kernel[8], 5, sizeof(cl_mem), &other[0].buffer);
            err |= clSetKernelArg(kernels::kernel[8], 6, sizeof(cl_float), &timestep);
            err |= clSetKernelArg(kernels::kernel[8], 7, sizeof(cl_mem), &this->buffer);
            err |= clSetKernelArg(kernels::kernel[8], 8, sizeof(cl_mem), &partC.buffer);
            err |= clSetKernelArg(kernels::kernel[8], 9, sizeof(cl_int), &n[0]);
            err |= clSetKernelArg(kernels::kernel[8], 10, sizeof(cl_int), &n[1]);
            err |= clSetKernelArg(kernels::kernel[8], 11, sizeof(cl_int), &N);
            
            err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel[8], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
            clFinish(kernels::queue);
            err |= clSetKernelArg(kernels::kernel_math[0], 0, sizeof(cl_mem), &partD.buffer);
            err |= clSetKernelArg(kernels::kernel_math[0], 2, sizeof(cl_mem), &partC.buffer);
            err |= clSetKernelArg(kernels::kernel_math[0], 1, sizeof(cl_mem), &this->buffer);
            err |= clSetKernelArg(kernels::kernel_math[0], 3, sizeof(cl_uint), &N);
            err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel_math[0], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        }

        else if(SP.getvalue<int>(Piro::SolveParams::TIMESCHEME) == 12){
            // std::cout << "Backward Euler" << std::endl;
                if(SP.getvalue<int>(Piro::SolveParams::SOLVERSCHEME) == 27){
                    Piro::logger::info("LU Decomposition");
                    if(RHS_INIT == false){
                        Piro::matrix_operations::lu_decomposition_HTLF(other);
                        RHS_INIT = true;
                    }
                    Piro::logger::debug("RHS_INIT end" );
                }
        }
        return partD;
    }

    CLBuffer operator*(CLBuffer partA, CLBuffer partB){
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernels::kernel_math[2], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernels::kernel_math[2], 1, sizeof(cl_mem), &partA.buffer);
        err |= clSetKernelArg(kernels::kernel_math[2], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernels::kernel_math[2], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel_math[2], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        
        return partC;
    }

    CLBuffer operator*(const float partA, CLBuffer partB){
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernels::kernel_math[6], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernels::kernel_math[6], 1, sizeof(cl_mem), &partA);
        err |= clSetKernelArg(kernels::kernel_math[6], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernels::kernel_math[6], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel_math[6], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator+(CLBuffer partA, CLBuffer partB){
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernels::kernel_math[0], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernels::kernel_math[0], 1, sizeof(cl_mem), &partA.buffer);
        err |= clSetKernelArg(kernels::kernel_math[0], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernels::kernel_math[0], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel_math[0], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator+(const float partA, CLBuffer partB){
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernels::kernel_math[4], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernels::kernel_math[4], 1, sizeof(cl_mem), &partA);
        err |= clSetKernelArg(kernels::kernel_math[4], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernels::kernel_math[4], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel_math[4], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }
        
    CLBuffer operator-(CLBuffer partA, CLBuffer partB){
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernels::kernel_math[1], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernels::kernel_math[1], 1, sizeof(cl_mem), &partA.buffer);
        err |= clSetKernelArg(kernels::kernel_math[1], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernels::kernel_math[1], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel_math[1], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator-(const float partA, CLBuffer partB){
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernels::kernel_math[5], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernels::kernel_math[5], 1, sizeof(cl_mem), &partA);
        err |= clSetKernelArg(kernels::kernel_math[5], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernels::kernel_math[5], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel_math[5], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator/(CLBuffer partA, CLBuffer partB){
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernels::kernel_math[3], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernels::kernel_math[3], 1, sizeof(cl_mem), &partA.buffer);
        err |= clSetKernelArg(kernels::kernel_math[3], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernels::kernel_math[3], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel_math[3], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator/(const float partA, CLBuffer partB){
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        err |= clSetKernelArg(kernels::kernel_math[7], 0, sizeof(cl_mem), &partC.buffer);
        err |= clSetKernelArg(kernels::kernel_math[7], 1, sizeof(cl_mem), &partA);
        err |= clSetKernelArg(kernels::kernel_math[7], 2, sizeof(cl_mem), &partB.buffer);
        err |= clSetKernelArg(kernels::kernel_math[7], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel_math[7], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }
};
