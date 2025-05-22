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
#include <kernelprocess.hpp>
#include <kernelmethods.hpp>

namespace Piro{
    CLBuffer CLBuffer::operator=(const std::vector<CLBuffer>& other){
        Piro::kernels& kernels = Piro::kernels::getInstance();
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        int nnz = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + MP.getvalue<int>(Piro::MeshParams::CONSTANTNUM)].values.size();
        int P = 1;
        std::vector<float>A(N, 0.0);
        CLBuffer partC, partD;
        partC.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);
        partD.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * N, A.data(), &err);
        
        Piro::SolveParams& SP = Piro::SolveParams::getInstance();
        float timestep = SP.getvalue<float>(Piro::SolveParams::TIMESTEP);

        if(SP.getvalue<int>(Piro::SolveParams::TIMESCHEME) == 11){
            // Forward Euler
            if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 0){
                Piro::kernelsolve::CSR::TIMESCHEME_11(other, N, P, n, timestep, partC, partD, 
                                                    this->buffer, 
                                                    this->buffer);
            }
            else if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 1){
                Piro::kernelsolve::DENSE::TIMESCHEME_11(other, N, P, timestep, this->buffer, partC, partD);

            }
            else if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 2){
                Piro::kernelsolve::HT::TIMESCHEME_11(other, N, SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE), this->buffer, partC, partD, timestep);

            }
            else if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 3){
                Piro::kernelsolve::COO::TIMESCHEME_11(other, N, nnz, n, timestep, partC, partD, this->buffer);

            }
            else{
                Piro::logger::info("Wrong data type");
                std::exit(1);
            }
        }

        else if(SP.getvalue<int>(Piro::SolveParams::TIMESCHEME) == 12){
            // Backward euler
            if(SP.getvalue<int>(Piro::SolveParams::SOLVERSCHEME) == 27){
                Piro::logger::info("LU Decomposition");
                if(INIT::getInstance().RHS_INIT == false){
                    if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 0){
                    }
                    else if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 1){
                    }
                    else if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 2){
                        Piro::matrix_operations::HT::lu_decomposition(other);
                    }
                    else if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 3){
                    }
                    else{
                        Piro::logger::info("Wrong data type");
                        std::exit(1);
                    }
                    INIT::getInstance().RHS_INIT = true;
                }
                Piro::logger::debug("RHS_INIT end" );
            }
        }
        return partD;
    }

    std::vector<CLBuffer> operator&(std::vector<CLBuffer> partA, std::vector<CLBuffer> partB){
         Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
        // part A is a vector, part B is a matrix
        Piro::kernelmethods::csrgeam(partA, partB);
        return {CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[0], 
                CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[1], 
                CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[2]};

    }

    CLBuffer operator*(CLBuffer partA, CLBuffer partB){
        Piro::kernels& kernels = Piro::kernels::getInstance();
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[2], 0, sizeof(cl_mem), &partC.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[2], 1, sizeof(cl_mem), &partA.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[2], 2, sizeof(cl_mem), &partB.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[2], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[2], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        
        return partC;
    }

    CLBuffer operator*(const float partA, CLBuffer partB){
        Piro::kernels& kernels = Piro::kernels::getInstance();
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[6], 0, sizeof(cl_mem), &partC.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[6], 1, sizeof(cl_mem), &partA);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[6], 2, sizeof(cl_mem), &partB.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[6], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[6], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator+(CLBuffer partA, CLBuffer partB){
        Piro::kernels& kernels = Piro::kernels::getInstance();
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 0, sizeof(cl_mem), &partC.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 1, sizeof(cl_mem), &partA.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 2, sizeof(cl_mem), &partB.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator+(const float partA, CLBuffer partB){
        Piro::kernels& kernels = Piro::kernels::getInstance();
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[4], 0, sizeof(cl_mem), &partC.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[4], 1, sizeof(cl_mem), &partA);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[4], 2, sizeof(cl_mem), &partB.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[4], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[4], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }
        
    CLBuffer operator-(CLBuffer partA, CLBuffer partB){
        Piro::kernels& kernels = Piro::kernels::getInstance();
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[1], 0, sizeof(cl_mem), &partC.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[1], 1, sizeof(cl_mem), &partA.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[1], 2, sizeof(cl_mem), &partB.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[1], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[1], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator-(const float partA, CLBuffer partB){
        Piro::kernels& kernels = Piro::kernels::getInstance();
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[5], 0, sizeof(cl_mem), &partC.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[5], 1, sizeof(cl_mem), &partA);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[5], 2, sizeof(cl_mem), &partB.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[5], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[5], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator/(CLBuffer partA, CLBuffer partB){
        Piro::kernels& kernels = Piro::kernels::getInstance();
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[3], 0, sizeof(cl_mem), &partC.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[3], 1, sizeof(cl_mem), &partA.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[3], 2, sizeof(cl_mem), &partB.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[3], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[3], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }

    CLBuffer operator/(const float partA, CLBuffer partB){
        Piro::kernels& kernels = Piro::kernels::getInstance();
        cl_int err;
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
        int N = n[0] * n[1] * n[2];
        std::vector<float>A(N, 0.0);
        CLBuffer partC;
        partC.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

        
        size_t globalWorkSize[1] = { (size_t)N };
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[7], 0, sizeof(cl_mem), &partC.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[7], 1, sizeof(cl_mem), &partA);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[7], 2, sizeof(cl_mem), &partB.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[7], 3, sizeof(cl_uint), &N);
        err = clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[7], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        return partC;
    }
};
