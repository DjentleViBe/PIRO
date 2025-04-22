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
#include <unordered_set>
#include <algorithm>
#include <cassert>

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
                        std::vector<float> Lap_full = {-6.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                        1.0,-6.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                                                        1.0, 0.0,-6.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                                                        0.0, 1.0, 1.0,-6.0, 0.0, 0.0, 0.0, 1.0,
                                                        1.0, 0.0, 0.0, 0.0,-6.0, 1.0, 1.0, 0.0,
                                                        0.0, 1.0, 0.0, 0.0, 1.0,-6.0, 0.0, 1.0,
                                                        0.0, 0.0, 1.0, 0.0, 1.0, 0.0,-6.0, 1.0,
                                                        0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,-6.0};
                        
                        int N = 8;
                        std::vector<float> Lap_val_V = {-6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1};
                        std::vector<int> Lap_ind_V = {0, 1, 2, 4, 1, 0, 3, 5, 2, 3, 0, 6, 3, 2, 1, 7, 4, 5, 6, 0, 5, 4, 7, 1, 6, 7, 4, 2, 7, 6, 5, 3};
                        std::vector<int> Lap_rowptr_V = {0, 4, 8, 12, 16, 20, 24, 28, 32};
                        float load = 0.7;
                        // int TABLE_SIZE = nextPowerOf2(Lap_ind_V.size() / load);
                        int TABLE_SIZE = Lap_ind_V.size() / load;
                        std::cout << "table size : " << TABLE_SIZE << std::endl;
                        std::vector<float> Hash_val_V(TABLE_SIZE);
                        std::vector<int> Hash_keys_V(TABLE_SIZE, -1);
                        int ind;
                        /////////////////////////////////////// Generate hash tables ///////////////////////////////////////////
                        for (int i = 0; i < N; i++){
                            int start = Lap_rowptr_V[i];
                            int end = Lap_rowptr_V[i + 1];
                            for(int j = start; j < end; j++){
                                int col = Lap_ind_V[j];
                                int row = i;

                                ind = row * N + col;
                                sethash(ind, Lap_val_V[j], TABLE_SIZE, Hash_keys_V, Hash_val_V);
                            }
                        }
                        std::cout << "hash keys : ";
                        printVector(Hash_keys_V);
                        std::cout << "hash val : ";
                        printVector(Hash_val_V);
                        std::cout << "Loop begin" << std::endl;
                        
                        /////////////////////////////////////////////////////////////////////////////////////////
                        
                        CLBuffer LFvalues, LFkeys;
                        cl_event event0, event1, event4, event5;
                        // cl_event event4;
                        LFvalues.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(float) * TABLE_SIZE, nullptr, &err);
                        LFkeys.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(int) * TABLE_SIZE, nullptr, &err);
                        err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &LFkeys.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &LFvalues.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_int), &N);
                        err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_int), &TABLE_SIZE);
                        print_time();
                        std::cout << "Loop begin" << std::endl;
                        //std::cout << "before Write buffer : ";
                        //printVector(Lap_val_V);
                        size_t globalWorkSize[1];
                        size_t localWorkSize[1];
                        std::cout << "write buffer start\n";
                        // clFinish(queue);
                        // std::cout << "rowouter = " << rowouter << ", Lap_rowptr_V.size() = " << Lap_rowptr_V.size() << std::endl;
                        err = clEnqueueWriteBuffer(queue, LFkeys.buffer, CL_FALSE, 
                                                    0, 
                                                    sizeof(int) * TABLE_SIZE,
                                                    Hash_keys_V.data(), 
                                                    0, nullptr, &event0);
                        // assert(Lap_rowptr_V.data() + rowouter != nullptr);
                        
                        err = clEnqueueWriteBuffer(queue, LFvalues.buffer, CL_FALSE, 
                                                    0, 
                                                    sizeof(float) * TABLE_SIZE,
                                                    Hash_val_V.data(), 
                                                    0, nullptr, &event1);
                        
                        // Wait for all transfers to complete
                        clWaitForEvents(2, (cl_event[]){event0, event1});
                        std::cout << "write buffer end\n";
                        // printVector(Lap_ind_V);
                        for (int rowouter = 0; rowouter < N - 1; rowouter++){
                            print_time();
                            std::cout << "rowouter : " << rowouter << std::endl;
                            std::vector<int> rowouter_cols;
                            // extract rowouter
                            for(int co = 0; co < N; co++){
                                float valofrow = lookup(rowouter, co, N, Hash_keys_V, Hash_val_V, TABLE_SIZE);
                                if(valofrow != 0){
                                    rowouter_cols.push_back(co);
                                }
                            }
                            
                            for (int r = rowouter + 1; r < N; ++r) {
                                if(lookup(r, rowouter, N, Hash_keys_V, Hash_val_V, TABLE_SIZE) == 0.0){
                                    continue;
                                }

                                for (int col : rowouter_cols) {
                                    // std::cout << col << ", ";
                                    // if(col < rowouter) continue;
                                    if(lookup(r, col, N, Hash_keys_V, Hash_val_V, TABLE_SIZE) == 0.0){
                                        // set the col in the hash table directly
                                        ind = r * N + col;
                                        // std::cout << "ind = " << ind << std::endl;
                                        sethash(ind, 0.0f, TABLE_SIZE, Hash_keys_V, Hash_val_V);
                                        // std::cout << col << ", ";
                                        
                                    }
                                }

                            }                                   
                            print_time();
                            printVector(Hash_keys_V);
                            printVector(Hash_val_V);
                            printVector(Lap_rowptr_V);
                            err = clEnqueueWriteBuffer(queue, LFkeys.buffer, CL_FALSE, 
                                                        0, 
                                                        sizeof(int) * TABLE_SIZE,
                                                        Hash_keys_V.data(), 
                                                        0, nullptr, &event0);
                            // assert(Lap_rowptr_V.data() + rowouter != nullptr);
    
                            err = clEnqueueWriteBuffer(queue, LFvalues.buffer, CL_FALSE, 
                                                        0, 
                                                        sizeof(float) * TABLE_SIZE,
                                                        Hash_val_V.data(), 
                                                        0, nullptr, &event1);

                            // Wait for all transfers to complete
                            clWaitForEvents(2, (cl_event[]){event0, event1});

                                                    
                            size_t nnz = (size_t)((N - rowouter - 1) * (N - rowouter));
                            size_t local = 2; // or whatever max workgroup size your device supports

                            globalWorkSize[0] = nnz;
                            localWorkSize[0] = local;
                            // std::cout << "after Write buffer : ";
                            
                            // std::cout << "Launching kernel" << std::endl;
                            err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_int), &rowouter);
                            err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
                            clFinish(queue);
                            //Hash_val_V.resize(TABLE_SIZE);
                            //Hash_keys_V.resize(TABLE_SIZE);
                            copyCL_offset<float>(queue, LFvalues.buffer, Hash_val_V, 0, TABLE_SIZE, &event4);
                            copyCL_offset<int>(queue, LFkeys.buffer, Hash_keys_V, 0, TABLE_SIZE, &event5);
                        
                          
                        }
                        print_time();
                        std::cout << "loop end" << std::endl;
                        // copyCL_offset<float>(queue, LFvalues.buffer, Hash_val_V, 0, TABLE_SIZE, &event4);
                        // copyCL_offset<int>(queue, LFkeys.buffer, Hash_keys_V, 0, TABLE_SIZE, &event5);
                        
                        printVector(Hash_keys_V);
                        printVector(Hash_val_V);
                        hash_to_dense_and_print(Hash_keys_V, Hash_val_V, N, TABLE_SIZE);
                        // printVector(Lap_col_V);
                        // printVector(Lap_rowptr_V);
                        // csr_to_dense_and_print(Lap_rowptr_V, Lap_ind_V, Lap_val_V, N);
                        clReleaseMemObject(LFvalues.buffer);
                        clReleaseMemObject(LFkeys.buffer);

                        // printVector(Lap_val_V);

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
    };
};

extern Giro::CellDataGPU CDGPU;

#endif // OPERATOR_OVERLOAD_HPP