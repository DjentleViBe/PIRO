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
#include <set>
#include <unordered_set>
#include <algorithm>

namespace Giro{
    class Equation{
        public:
            cl_mem operandvalues;
            cl_mem operandcolumns;
            cl_mem operandrowptr;
            cl_mem operandrows;
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
                        Logger::info("LU Decomposition");
                        int N = MP.n[0] * MP.n[1] * MP.n[2];
                        int index = MP.vectornum + MP.scalarnum;
                        if(RHS_INIT == false){
                            cl_event event0, event1, event4, event5, event7, event8, event9;
                            Logger::debug("Buffer creation begin");
                            
                            MP.AMR[0].CD[index].rowpointers = copyCL<int>(queue, other[0].buffer, N + 1, &event7);
                            MP.AMR[0].CD[index].columns = copyCL<int>(queue, other[1].buffer, RHS.sparsecount, &event8);
                            MP.AMR[0].CD[index].values = copyCL<float>(queue, other[2].buffer, RHS.sparsecount, &event9);

                            
                            /////////////////////////////////////////////////////////////////////////////////////////
                            auto& cd = MP.AMR[0].CD[index];
                            CLBuffer LFvalues, LFkeys;
                            float load = SP.a * pow(N, SP.b) + SP.c;
                            // int TABLE_SIZE = nextPowerOf2(cd.columns.size() / load);
                            // int raw_size = cd.columns.size() / load;
                            // int TABLE_SIZE = next_prime(raw_size);
                            int TABLE_SIZE = RHS.sparsecount / (load * SP.loadfactor);
                            Logger::debug("RHS_INIT begin, sparse count :", RHS.sparsecount, ", Table size :" , TABLE_SIZE, ", Load factor :", load * SP.loadfactor);
                            Logger::warning("N * N :", N*N);
                            
                            LFvalues.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(float) * TABLE_SIZE, nullptr, &err);
                            LFkeys.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(int) * TABLE_SIZE, nullptr, &err);
                            err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &LFkeys.buffer);
                            err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &LFvalues.buffer);
                            err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_int), &N);
                            err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_int), &TABLE_SIZE);
                            Logger::debug("Buffer creation end");
                            size_t globalWorkSize[1];
                            size_t localWorkSize[1];
                            int ind;
                            Logger::debug("Loop begin");
                            
                            std::vector<float> Hash_val_V(TABLE_SIZE);
                            std::vector<int> Hash_keys_V(TABLE_SIZE, -1);
                            /////////////////////////////////////// Generate hash tables ///////////////////////////////////////////
                            for (int i = 0; i < N; i++){
                                int start = cd.rowpointers[i];
                                int end = cd.rowpointers[i + 1];
                                for(int j = start; j < end; j++){
                                    int col = cd.columns[j];
                                    int row = i;

                                    ind = row * N + col;
                                    sethash(ind, cd.values[j], TABLE_SIZE, Hash_keys_V, Hash_val_V);
                                }
                            }
                            // hash_to_dense_and_print(Hash_keys_V, Hash_val_V, N, TABLE_SIZE);
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
                            int limit;
                            for (int rowouter = 0; rowouter < N; rowouter++){
                                Logger::info("rowouter :", rowouter, ", HashTable size :", TABLE_SIZE);
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
                                Logger::debug("Inserting 0s finished");
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
                                Logger::debug("Map memory object finished");
                                // std::cout << cd.rowpointers[N] << std::endl;
                                limit = (N - rowouter - 1) * (N - rowouter);
                                size_t nnz = (size_t)limit;
                                size_t local = (size_t)maxWorkGroupSize; // or whatever max workgroup size your device supports
                                Logger::warning("nnz :", nnz);
                                globalWorkSize[0] = ((nnz + local - 1) / local) * local;
                                localWorkSize[0] = local;
                                // size_t localWorkSize[1] = { globalWorkSize[0] / 4 };
                                err |= clSetKernelArg(kernelfilterarray, 5, sizeof(cl_int), &limit);
                                err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_int), &rowouter);
                                err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
                                clFinish(queue);
                                Logger::debug("Kernel finished");
                                copyCL_offset<float>(queue, LFvalues.buffer, Hash_val_V, 0, TABLE_SIZE, &event4);
                                copyCL_offset<int>(queue, LFkeys.buffer, Hash_keys_V, 0, TABLE_SIZE, &event5);
                                Logger::debug("CopyCL");
                                Logger::debug("Erased");
                                // hash_to_dense_and_print(Hash_keys_V, Hash_val_V, N, TABLE_SIZE);
                                
                            }
                            // copyCL_offset<float>(queue, LFvalues.buffer, Hash_val_V, 0, TABLE_SIZE, &event4);
                            // copyCL_offset<int>(queue, LFkeys.buffer, Hash_keys_V, 0, TABLE_SIZE, &event5);    
                            Logger::debug("loop end");
                            // printVector(MP.AMR[0].CD[index].values);
                            // printVector(MP.AMR[0].CD[index].rowpointers);
                            // csr_to_dense_and_print(cd.rowpointers,
                            //                         cd.columns, 
                            //                        cd.values, N);
                            RHS_INIT = true;
                            hash_to_dense_and_print(Hash_keys_V, Hash_val_V, N, TABLE_SIZE);
                        }
                        Logger::debug("RHS_INIT end" );
                        // csr_to_dense_and_print(MP.AMR[0].CD[index].rowpointers, MP.AMR[0].CD[index].columns, MP.AMR[0].CD[index].values, N);
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