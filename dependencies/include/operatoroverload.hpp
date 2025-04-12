// operator_overload.hpp

#ifndef OPERATOROVERLOADHPP
#define OPERATOROVERLOADHPP
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#elif _WIN32
    #include "./CL/opencl.h"
#else
    #include "./CL/opencl.h"
#endif
#include "preprocess.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>

namespace Giro{
    class Equation{
        public:
            cl_mem operandvalues;
            cl_mem operandcolumns;
            cl_mem operandrowptr;
            int sparsecount;
    };
};

extern Giro::Equation RHS;
const float EPSILON = 1E-6;

class CLBuffer{
    public:
        cl_mem buffer;
        
        CLBuffer operator=(const std::vector<CLBuffer>& other){

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
                    if(SP.solverscheme == 17){
                        std::cout << "LU Decomposition" << std::endl;
                        int N = MP.n[0] * MP.n[1] * MP.n[2];
                        int factor = N * N * 3 / 4;
                        int index = MP.vectornum + MP.scalarnum;
                        print_time();
                        std::cout << "RHS_INIT begin, sparse count : " << RHS.sparsecount << std::endl;
                        if(RHS_INIT == false){
                            // int TABLE_SIZE = nextPowerOf2(N);
                            std::vector<float> Value_filtered_E(N * N);
                            CLBuffer Value_filtered;
                            cl_event event6, event7, event8, event9;
                            print_time();
                            std::cout << "Buffer creation begin" << std::endl;
                            // calculate remaining memory
                            std::cout << "Total variable count : " << MP.scalarnum + MP.vectornum << std::endl;
                            std::cout << "Total memory : " << globalMemSize << std::endl;
                            std::cout << "Max alloc : " << maxAllocSize << std::endl;

                            cl_ulong rem = globalMemSize - sizeof(float) * (2 * N + RHS.sparsecount) - sizeof(int) * (2 * N + 1 + RHS.sparsecount);
                            std::cout << "remaining : " << rem << std::endl;
                            cl_ulong check = (sizeof(int) + sizeof(float)) * (N * N);
                            if( check < rem){
                                std::cout << "One pass" << std::endl;
                                // check if N*N is passing for max alloc
                                if(check < maxAllocSize){
                                    std::cout << "Buffer size within Max Allocoation size" << std::endl;
                                }
                            }
                            
                            Value_filtered.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(float) * N * N, nullptr, &err);
                        
                            print_time();
                            std::cout << "Buffer creation end" << std::endl;
                        
                            size_t globalWorkSize[1] = { (size_t)N * N};
                            // size_t local_work_size  = 256;           
                            float fillValue = 0.0f;
                            int fillValue_int = 0;
                            RHS.operandrowptr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * (N + 1), nullptr, &err);
                            RHS.operandcolumns = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * factor, nullptr, &err);
                            RHS.operandvalues = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * factor, nullptr, &err);
                            clEnqueueFillBuffer(queue, RHS.operandvalues, &fillValue, sizeof(float), 0, sizeof(float) *factor, 0, nullptr, nullptr);
                            clEnqueueFillBuffer(queue, RHS.operandcolumns, &fillValue_int, sizeof(int), 0, sizeof(int) *factor, 0, nullptr, nullptr);
                            clEnqueueFillBuffer(queue, RHS.operandrowptr, &fillValue_int, sizeof(int), 0, sizeof(int) * (N + 1), 0, nullptr, nullptr);
                            clFinish(queue);
                            clEnqueueCopyBuffer(queue, other[2].buffer, RHS.operandvalues, 0, 0, sizeof(float) * RHS.sparsecount, 0, NULL, NULL);
                            clEnqueueCopyBuffer(queue, other[1].buffer, RHS.operandcolumns, 0, 0, sizeof(int) * RHS.sparsecount, 0, NULL, NULL);
                            clEnqueueCopyBuffer(queue, other[0].buffer, RHS.operandrowptr, 0, 0, sizeof(int) * (N + 1), 0, NULL, NULL);
                            clFinish(queue);
                            MP.AMR[0].CD[index].values = copyCL<float>(queue, other[2].buffer, RHS.sparsecount, &event7);
                            MP.AMR[0].CD[index].columns = copyCL<int>(queue, other[1].buffer, RHS.sparsecount, &event8);
                            MP.AMR[0].CD[index].rowpointers = copyCL<int>(queue, other[0].buffer, N + 1, &event9);
                            // int N_square = N * N;
                            err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &RHS.operandvalues);
                            err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &RHS.operandcolumns);
                            err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_mem), &RHS.operandrowptr);
                            err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_mem), &Value_filtered.buffer);
                            err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_int), &N);
                            err |= clSetKernelArg(kernelfilterarray, 5, sizeof(cl_int), &RHS.sparsecount);
                            clWaitForEvents(3, (cl_event[]){event7, event8, event9});
                            printVector(MP.AMR[0].CD[index].rowpointers);
                            print_time();
                            std::cout << "Loop begin" << std::endl;
                            for (int rowouter = 0; rowouter < 1; rowouter++){
                                print_time();
                                std::cout << "rowouter : " << rowouter << std::endl;
                                err |= clSetKernelArg(kernelfilterarray, 6, sizeof(cl_int), &rowouter);
                                    
                                err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize, 0, 0, NULL, NULL);
                                clFinish(queue);
                                
                                Value_filtered_E = copyCL<float>(queue, Value_filtered.buffer, N*N, &event6);
                                clWaitForEvents(1, (cl_event[]){event6});
                                printVector(Value_filtered_E);

                                    
                                
                            }
                            print_time();
                            std::cout << "loop end" << std::endl;
                            clReleaseMemObject(Value_filtered.buffer);
                            RHS_INIT = true;
                        }
                        print_time();
                        std::cout << "RHS_INIT end" << std::endl;
                        // printVector(MP.AMR[0].CD[index].values);
                    }
            }

            return partD;

            
        }

        friend CLBuffer operator*(CLBuffer partA, CLBuffer partB){
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

        friend CLBuffer operator*(const float partA, CLBuffer partB){
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

        friend CLBuffer operator+(CLBuffer partA, CLBuffer partB){
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

        friend CLBuffer operator+(const float partA, CLBuffer partB){
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
        
        friend CLBuffer operator-(CLBuffer partA, CLBuffer partB){
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

        friend CLBuffer operator-(const float partA, CLBuffer partB){
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

        friend CLBuffer operator/(CLBuffer partA, CLBuffer partB){
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

        friend CLBuffer operator/(const float partA, CLBuffer partB){
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

namespace Giro{
    class CellDataGPU{
        public:
            std::vector<CLBuffer> values_gpu;
            std::vector<CLBuffer> laplacian_csr;
            std::vector<CLBuffer> gradient;
    };
};

extern Giro::CellDataGPU CDGPU;

#endif // OPERATOR_OVERLOAD_HPP