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
                        std::cout << "RHS_INIT begin" << std::endl;
                        if(RHS_INIT == false){
                            int TABLE_SIZE = nextPowerOf2(N);
                            std::vector<float> Value_filtered_E(N);
                            CLBuffer Value_filtered;
                            CLBuffer hk_0, hv_0, hk_r, hv_r;
                            cl_event event1, event2, event3, event4, event5, event6, event7, event8, event9;
                            print_time();
                            std::cout << "Buffer creation begin" << std::endl;
                            
                            // int row = 0;
                            // float sizefactor = 2 / 3 ;
                            hk_0.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(int) * N, nullptr, &err);
                            hv_0.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * N, nullptr, &err);
                            hk_r.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(int) * N, nullptr, &err);
                            hv_r.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(float) * N, nullptr, &err);
                            
                            Value_filtered.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(float) * N, nullptr, &err);
                        
                            print_time();
                            std::cout << "Buffer creation end" << std::endl;
                        
                            err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &hk_0.buffer);
                            err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &hv_0.buffer);
                            err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_mem), &hk_r.buffer);
                            err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_mem), &hv_r.buffer);
                            err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_mem), &Value_filtered.buffer);
                            err |= clSetKernelArg(kernelfilterarray, 5, sizeof(cl_int), &N);
                            size_t globalWorkSize_square[1] = { (size_t)N};
                            std::vector<float> hashvalues_0(TABLE_SIZE, 0.0f);
                            std::vector<int> hashkeys_0(TABLE_SIZE, -1);
                            std::vector<float> hashvalues_r(TABLE_SIZE, 0.0f);
                            std::vector<int> hashkeys_r(TABLE_SIZE, -1);
                            int* hk_0_ptr = (int*)clEnqueueMapBuffer(queue, hk_0.buffer, CL_FALSE, CL_MAP_WRITE, 0, sizeof(int) * hashkeys_0.size(), 0, nullptr, nullptr, &err);
                            float* hv_0_ptr = (float*)clEnqueueMapBuffer(queue, hv_0.buffer, CL_FALSE, CL_MAP_WRITE, 0, sizeof(float) * hashvalues_0.size(), 0, nullptr, nullptr, &err);
                            int* hk_r_ptr = (int*)clEnqueueMapBuffer(queue, hk_r.buffer, CL_FALSE, CL_MAP_WRITE, 0, sizeof(int) * hashkeys_r.size(), 0, nullptr, nullptr, &err);
                            float* hv_r_ptr = (float*)clEnqueueMapBuffer(queue, hv_r.buffer, CL_FALSE, CL_MAP_WRITE, 0, sizeof(float) * hashvalues_r.size(), 0, nullptr, nullptr, &err);
                            
                            float* VE = (float*)clEnqueueMapBuffer(queue, Value_filtered.buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * Value_filtered_E.size(), 0, nullptr, nullptr, &err);
                            float fillValue = 0.0f;
                            int fillValue_int = 0;
                            // std::cout << N << std::endl;
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
                            clWaitForEvents(3, (cl_event[]){event7, event8, event9});
                            print_time();
                            std::cout << "Loop begin" << std::endl;
                            for (int rowouter = 0; rowouter < N; rowouter++){
                                err |= clSetKernelArg(kernelfilterarray, 6, sizeof(cl_int), &rowouter);
                                std::fill(hashvalues_0.begin(), hashvalues_0.end(), 0.0f);
                                std::fill(hashkeys_0.begin(), hashkeys_0.end(), -1);
                                /////////////// generate hash table for the 0th row
                                for (int j = MP.AMR[0].CD[index].rowpointers[rowouter]; j < MP.AMR[0].CD[index].rowpointers[rowouter + 1]; j++) {
                                    int key_0 = MP.AMR[0].CD[index].columns[j]; // Original index
                                    int hashindex_0 = key_0 & (TABLE_SIZE - 1); 
                                    int startindex = hashindex_0;

                                    while (true) {  
                                        int existing_key = hashkeys_0[hashindex_0];
                                
                                        if (existing_key == -1 || existing_key == key_0) {  // Found empty or same key
                                            hashkeys_0[hashindex_0] = key_0;
                                            hashvalues_0[hashindex_0] = MP.AMR[0].CD[index].values[j];
                                            break;
                                        }
                                
                                        hashindex_0 = (hashindex_0 + 1) & (TABLE_SIZE - 1);  // Faster wrapping
                                
                                        if (hashindex_0 == startindex) {  // Table full
                                            std::cerr << "Error: Hash table is full, cannot insert key " << key_0 << std::endl;
                                            break;
                                        }
                                    }

                                }
                                std::memcpy(hk_0_ptr, hashkeys_0.data(), sizeof(int) * hashkeys_0.size());
                                std::memcpy(hv_0_ptr, hashvalues_0.data(), sizeof(float) * hashvalues_0.size());
                                err = clEnqueueUnmapMemObject(queue, hk_0.buffer, hk_0_ptr, 0, nullptr, &event1);
                                err = clEnqueueUnmapMemObject(queue, hv_0.buffer, hv_0_ptr, 0, nullptr, &event2);

                                for (int row = rowouter + 1; row < N; row ++){
                                    clEnqueueFillBuffer(queue, Value_filtered.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N, 0, nullptr, &event5);
                                    std::fill(hashvalues_r.begin(), hashvalues_r.end(), 0.0f);
                                    std::fill(hashkeys_r.begin(), hashkeys_r.end(), -1);
                                    // populate rth row hash //////////////////////////
                                    for (int j = MP.AMR[0].CD[index].rowpointers[row]; j < MP.AMR[0].CD[index].rowpointers[row + 1]; j++) {
                                        int key_r = MP.AMR[0].CD[index].columns[j]; // Original index
                                        //int hashindex = (int)(TABLE_SIZE * (key * A - (int)(key * A))) % TABLE_SIZE;
                                        int hashindex_r = key_r & (TABLE_SIZE - 1);
                                        int startindex = hashindex_r;
                                        // bool inserted = false;
        
                                        while (true) {  
                                            int existing_key = hashkeys_r[hashindex_r];
                                    
                                            if (existing_key == -1 || existing_key == key_r) {  // Found empty or same key
                                                hashkeys_r[hashindex_r] = key_r;
                                                hashvalues_r[hashindex_r] = MP.AMR[0].CD[index].values[j];
                                                break;
                                            }
                                    
                                            hashindex_r = (hashindex_r + 1) & (TABLE_SIZE - 1);  // Faster wrapping
                                    
                                            if (hashindex_r == startindex) {  // Table full
                                                std::cerr << "Error: Hash table is full, cannot insert key " << key_r << std::endl;
                                                break;
                                            }
                                        }
                                    }
                                    // print_time();
                                    // std::cout << "write buffer start" << std::endl;
                                    std::memcpy(hk_r_ptr, hashkeys_r.data(), sizeof(int) * hashkeys_r.size());
                                    std::memcpy(hv_r_ptr, hashvalues_r.data(), sizeof(float) * hashvalues_r.size());
                                    err = clEnqueueUnmapMemObject(queue, hk_r.buffer, hk_r_ptr, 0, nullptr, &event3);
                                    err = clEnqueueUnmapMemObject(queue, hv_r.buffer, hv_r_ptr, 0, nullptr, &event4);
        
                                    clWaitForEvents(5, (cl_event[]){event1, event2, event3, event4, event5, event6});
                                    
                                    // print_time();
                                    // std::cout << "write buffer end" << std::endl;
                                    err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize_square, NULL, 0, NULL, NULL);
                                    clFinish(queue);
                                    
                                    // std::memcpy(Value_filtered_E.data(), VE, sizeof(float) * Value_filtered_E.size());
                                    err = clEnqueueUnmapMemObject(queue, Value_filtered.buffer, VE, 0, nullptr, nullptr);
                                    clFinish(queue);
                                    if (err != CL_SUCCESS) {
                                        std::cerr << "Error in unmapping buffer: " << err << std::endl;
                                        
                                    }
                                    
                                    //for (size_t i = 0; i < Value_filtered_E.size(); ++i) {
                                    //    std::cout << VE[i] << " ";
                                    //    // Use the value as needed
                                    //}
                                    //std::cout << ""<< std::endl;
                                    Value_filtered_E = copyCL<float>(queue, Value_filtered.buffer, N, &event6);
                                    // printVector(Value_filtered_E);
                                    //printCL(Value_filtered.buffer, N, 1);
                                    std::vector<std::pair<int, double>> to_insert;
                                    std::vector<int> to_delete;
                                    int start = MP.AMR[0].CD[index].rowpointers[row];
                                    int end = MP.AMR[0].CD[index].rowpointers[row + 1];

                                    // Use a hash map for fast index lookup
                                    std::unordered_map<int, int> index_map;
                                    for (int r = start; r < end; ++r) {
                                        index_map[MP.AMR[0].CD[index].columns[r]] = r;
                                    }
                                    // clWaitForEvents(1, (cl_event[]){event6});
                                    
                                    for (int vf = 0; vf < N; vf++){
                                        double val = Value_filtered_E[vf];
                                        auto it = index_map.find(vf);
                                        // it = -1;
                                        if (std::abs(val) > 1e-6) {
                                            if (it != index_map.end()) {
                                                // Index exists, just update the value
                                                MP.AMR[0].CD[index].values[it->second] = val;
                                            } else {
                                                // New value, needs insertion
                                                to_insert.emplace_back(vf, val);
                                            }
                                        } else if (it != index_map.end()) {
                                            // Value is zero but index exists -> deletion
                                            to_delete.push_back(it->second);
                                        }
                                    }
                                    /* batch insertions*/
                                    for(int k = 0; k < to_insert.size();k++){
                                        
                                        int insert_pos = MP.AMR[0].CD[index].rowpointers[row + 1];
                                        MP.AMR[0].CD[index].columns.insert(MP.AMR[0].CD[index].columns.begin() + insert_pos, to_insert[k].first);
                                        MP.AMR[0].CD[index].values.insert(MP.AMR[0].CD[index].values.begin() + insert_pos, to_insert[k].second);
                                        
                                    }

                                    /* Batch deletions: erase from the back to avoid index shifting */
                                    std::sort(to_delete.rbegin(), to_delete.rend());
                                    for (int idx : to_delete) {
                                        MP.AMR[0].CD[index].columns.erase(MP.AMR[0].CD[index].columns.begin() + idx);
                                        MP.AMR[0].CD[index].values.erase(MP.AMR[0].CD[index].values.begin() + idx);
                                    }
                                    
                                    
                                    // Update row pointer offsets for future rows
                                    int delta = to_insert.size() - to_delete.size();
                                    if (delta != 0) {
                                        for (int k = row; k < N +1; k++) {
                                            MP.AMR[0].CD[index].rowpointers[k + 1] += delta;
                                        }
                                    }
                                }
                            }
                            print_time();
                            std::cout << "loop end" << std::endl;
                            err = clEnqueueUnmapMemObject(queue, hk_0.buffer, hk_0_ptr, 0, nullptr, nullptr);
                            err = clEnqueueUnmapMemObject(queue, hv_0.buffer, hv_0_ptr, 0, nullptr, nullptr);
                            err = clEnqueueUnmapMemObject(queue, hk_r.buffer, hk_r_ptr, 0, nullptr, nullptr);
                            err = clEnqueueUnmapMemObject(queue, hv_r.buffer, hv_r_ptr, 0, nullptr, nullptr);
                            clReleaseMemObject(Value_filtered.buffer);
                            clReleaseMemObject(hk_0.buffer);
                            clReleaseMemObject(hk_r.buffer);
                            clReleaseMemObject(hv_0.buffer);
                            clReleaseMemObject(hv_r.buffer);
                            RHS_INIT = true;
                        }
                        print_time();
                        std::cout << "RHS_INIT end" << std::endl;
                        printVector(MP.AMR[0].CD[index].values);
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