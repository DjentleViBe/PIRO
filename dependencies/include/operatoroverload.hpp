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
                        int N = 8;
                        
                        // size_t globalWorkSize[1] = { (size_t)N};
                        std::vector<float> Lap_full = {-6.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                        1.0,-6.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                                                        1.0,0.0,-6.0,1.0,0.0,0.0,1.0,0.0,
                                                        0.0,1.0,1.0,-6.0,0.0,0.0,0.0,1.0,
                                                        1.0,0.0,0.0,0.0,-6.0,1.0,1.0,0.0,
                                                        0.0,1.0,0.0,0.0,1.0,-6.0,0.0,1.0,
                                                        0.0,0.0,1.0,0.0,1.0,0.0,-6.0,1.0,
                                                        0.0,0.0,0.0,1.0,0.0,1.0,1.0,-6.0};
                        
                        
                        std::vector<float> Lap_val_V = {-6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1};
                        std::vector<int> Lap_ind_V = {0, 1, 2, 4, 1, 0, 3, 5, 2, 3, 0, 6, 3, 2, 1, 7, 4, 5, 6, 0, 5, 4, 7, 1, 6, 7, 4, 2, 7, 6, 5, 3};
                        std::vector<int> Lap_rowptr_V = {0, 4, 8, 12, 16, 20, 24, 28, 32};
                        int TABLE_SIZE = nextPowerOf2(N);
                        std::vector<float> Value_filtered_E = {0, 0, 0, 0, 0, 0, 0, 0};
                        CLBuffer Value_filtered;
                        CLBuffer hk_0, hv_0, hk_r, hv_r;
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
                        float fillValue = 0.0f;
                        int fillValue_int = -1;
                        print_time();
                        std::cout << "Loop begin" << std::endl;
                        // const float A = 0.6180339887;
                        for (int rowouter = 0; rowouter < N; rowouter++){
                            err |= clSetKernelArg(kernelfilterarray, 6, sizeof(cl_int), &rowouter);
                            std::vector<float> hashvalues_0(TABLE_SIZE, 0.0f);
                            std::vector<int> hashkeys_0(TABLE_SIZE, -1);
                            /////////////// generate hash table for the 0th row
                            // std::cout << "Generating hash table" << std::endl;
                            //print_time();
                            //std::cout << "starthash_0" << std::endl;
                            for (int j = Lap_rowptr_V[rowouter]; j < Lap_rowptr_V[rowouter + 1]; j++) {
                                int key_0 = Lap_ind_V[j]; // Original index
                                //int hashindex = (int)(TABLE_SIZE * (key * A - (int)(key * A))) % TABLE_SIZE;
                                int hashindex_0 = key_0 & (TABLE_SIZE - 1); 
                                int startindex = hashindex_0;
                                // bool inserted = false;

                                while (true) {  
                                    int existing_key = hashkeys_0[hashindex_0];
                            
                                    if (existing_key == -1 || existing_key == key_0) {  // Found empty or same key
                                        hashkeys_0[hashindex_0] = key_0;
                                        hashvalues_0[hashindex_0] = Lap_val_V[j];
                                        break;
                                    }
                            
                                    hashindex_0 = (hashindex_0 + 1) & (TABLE_SIZE - 1);  // Faster wrapping
                            
                                    if (hashindex_0 == startindex) {  // Table full
                                        std::cerr << "Error: Hash table is full, cannot insert key " << key_0 << std::endl;
                                        break;
                                    }
                                }
                            }
                            //print_time();
                            //std::cout << "endhash_0" << std::endl;
                            /////////////// generate hash table for the 0th row
                            clEnqueueFillBuffer(queue, hv_0.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N, 0, nullptr, nullptr);
                            clEnqueueFillBuffer(queue, hk_0.buffer, &fillValue_int, sizeof(int), 0, sizeof(int) * N, 0, nullptr, nullptr);
                            clFinish(queue);
                            err = clEnqueueWriteBuffer(queue, hk_0.buffer, CL_TRUE, 0, sizeof(int) * hashkeys_0.size(), hashkeys_0.data(), 0, NULL, NULL);
                            err = clEnqueueWriteBuffer(queue, hv_0.buffer, CL_TRUE, 0, sizeof(float) * hashvalues_0.size(), hashvalues_0.data(), 0, NULL, NULL);
                            clFinish(queue);
                            // printVector(hashkeys_0);
                            // std::cout << "hash values 0 : ";
                            // printVector(hashvalues_0);
                            // std::cout << "Values size : " << Lap_val_V.size() << std::endl;
                            for (int row = rowouter + 1; row < N; row ++){
                                std::vector<float> hashvalues_r(TABLE_SIZE, 0.0f);
                                std::vector<int> hashkeys_r(TABLE_SIZE, -1);
                                clEnqueueFillBuffer(queue, Value_filtered.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N, 0, nullptr, nullptr);
                                clFinish(queue);
                                //print_time();
                                //std::cout << "starthash_r" << std::endl;
                                // populate rth row hash //////////////////////////
                                for (int j = Lap_rowptr_V[row]; j < Lap_rowptr_V[row + 1]; j++) {
                                    int key_r = Lap_ind_V[j]; // Original index
                                    //int hashindex = (int)(TABLE_SIZE * (key * A - (int)(key * A))) % TABLE_SIZE;
                                    int hashindex_r = key_r & (TABLE_SIZE - 1);
                                    int startindex = hashindex_r;
                                    // bool inserted = false;
    
                                    while (true) {  
                                        int existing_key = hashkeys_r[hashindex_r];
                                
                                        if (existing_key == -1 || existing_key == key_r) {  // Found empty or same key
                                            hashkeys_r[hashindex_r] = key_r;
                                            hashvalues_r[hashindex_r] = Lap_val_V[j];
                                            break;
                                        }
                                
                                        hashindex_r = (hashindex_r + 1) & (TABLE_SIZE - 1);  // Faster wrapping
                                
                                        if (hashindex_r == startindex) {  // Table full
                                            std::cerr << "Error: Hash table is full, cannot insert key " << key_r << std::endl;
                                            break;
                                        }
                                    }
                                }
                                
                                clEnqueueFillBuffer(queue, hv_r.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N, 0, nullptr, nullptr);
                                clEnqueueFillBuffer(queue, hk_r.buffer, &fillValue_int, sizeof(int), 0, sizeof(int) * N, 0, nullptr, nullptr);
                                clFinish(queue);
                                err = clEnqueueWriteBuffer(queue, hk_r.buffer, CL_TRUE, 0, sizeof(int) * hashkeys_r.size(), hashkeys_r.data(), 0, NULL, NULL);
                                err = clEnqueueWriteBuffer(queue, hv_r.buffer, CL_TRUE, 0, sizeof(float) * hashvalues_r.size(), hashvalues_r.data(), 0, NULL, NULL);
                                clFinish(queue);
                                //print_time();
                                //std::cout << "startkernel" << std::endl;
                                err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize_square, NULL, 0, NULL, NULL);
                                clFinish(queue);
                                //print_time();
                                //std::cout << "endkernel" << std::endl;
                                // printCL(Value_filtered.buffer, N, 1);
                                
                                Value_filtered_E = copyCL(Value_filtered.buffer, N, 1);
                                // printVector(Value_filtered_E);
                                std::vector<std::pair<int, double>> to_insert;
                                std::vector<int> to_delete;
                                int start = Lap_rowptr_V[row];
                                int end = Lap_rowptr_V[row + 1];
                                int it = 0;
                                
                                for (int vf = 0; vf < N; vf++){
                                    double val = Value_filtered_E[vf];
                                    it = -1;
                                    for(int r = start; r < end; r++){
                                        if(Lap_ind_V[r] == vf){
                                            // index exists
                                            it = r;
                                            break;
                                        }
                                    }
                                    if (std::abs(val) > 1e-6){
                                        if(it != -1 && val != 0){
                                            // index exists and value exists, just replace value
                                            Lap_val_V[it] = val;
                                            //std::cout << "edits " << it << std::endl;
                                        }
                                        else if(it == -1 && val != 0){
                                            // index does not exist, value exists, insertions
                                            //std::cout << "insertions " << it << ", " << val << std::endl;
                                            to_insert.emplace_back(vf, val);
                                        }
                                    }
                                    else if(it != -1){
                                        // index exist, value does not exists, deletions
                                        // std::cout << "deletions " << it << std::endl;
                                        to_delete.push_back(it);
                                    }
                                }
                                /* batch insertions*/
                                for(int k = 0; k < to_insert.size();k++){
                                    
                                    int insert_pos = Lap_rowptr_V[row + 1];
                                    Lap_ind_V.insert(Lap_ind_V.begin() + insert_pos, to_insert[k].first);
                                    Lap_val_V.insert(Lap_val_V.begin() + insert_pos, to_insert[k].second);
                                    
                                }

                                /* Batch deletions: erase from the back to avoid index shifting */
                                std::sort(to_delete.rbegin(), to_delete.rend());
                                for (int idx : to_delete) {
                                    Lap_ind_V.erase(Lap_ind_V.begin() + idx);
                                    Lap_val_V.erase(Lap_val_V.begin() + idx);
                                }
                                
                                
                                // Update row pointer offsets for future rows
                                int delta = to_insert.size() - to_delete.size();
                                if (delta != 0) {
                                    for (int k = row; k < N +1; k++) {
                                        Lap_rowptr_V[k + 1] += delta;
                                    }
                                }
                                // printVector(Lap_val_V);
                                //printVector(Lap_rowptr_V);
                            }
                            // printVector(Lap_val_V);
                            //printVector(Lap_rowptr_V);
                        }
                            // std::cout << "\n";
                        print_time();
                        std::cout << "loop end" << std::endl;
                        
                        clReleaseMemObject(Value_filtered.buffer);
                        clReleaseMemObject(hk_0.buffer);
                        clReleaseMemObject(hk_r.buffer);
                        clReleaseMemObject(hv_0.buffer);
                        clReleaseMemObject(hv_r.buffer);
                        printVector(Lap_val_V);
                        // printVector(Lap_ind_V);

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