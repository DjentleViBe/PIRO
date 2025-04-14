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
                        std::vector<float> sorted_vals;
                        std::vector<int> sorted_indices;
                        cl_event event1, event2, event3;
                        // Loop through each row
                        for (int row = 0; row < Lap_rowptr_V.size() - 1; ++row) {
                            int start = Lap_rowptr_V[row];
                            int end = Lap_rowptr_V[row + 1];
                    
                            // Extract the elements of the current row
                            std::vector<int> row_indices(Lap_ind_V.begin() + start, Lap_ind_V.begin() + end);
                            std::vector<float> row_values(Lap_val_V.begin() + start, Lap_val_V.begin() + end);
                    
                            // Sort the row by column indices
                            std::vector<std::pair<int, float>> row_data;
                            for (size_t i = 0; i < row_indices.size(); ++i) {
                                row_data.push_back({row_indices[i], row_values[i]});
                            }
                    
                            std::sort(row_data.begin(), row_data.end());  // Sort by column index
                    
                            // Update sorted_vals and sorted_indices
                            for (const auto& pair : row_data) {
                                sorted_indices.push_back(pair.first);
                                sorted_vals.push_back(pair.second);
                            }
                        }
                    
                        // Now we have the sorted values and indices
                        Lap_val_V = sorted_vals;
                        Lap_ind_V = sorted_indices;
                        
                        /////////////////////////////////////////////////////////////////////////////////////////
                        
                        CLBuffer LFvalues, Lap_ind, Lap_rowptr;
                        LFvalues.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N * N, nullptr, &err);
                        Lap_ind.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * N * N, nullptr, &err);
                        Lap_rowptr.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * (N + 1), nullptr, &err);
                        err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &Lap_rowptr.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &Lap_ind.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_mem), &LFvalues.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_int), &N);
                        
                        print_time();
                        std::cout << "Loop begin" << std::endl;
                        //std::cout << "before Write buffer : ";
                        //printVector(Lap_val_V);
                        size_t globalWorkSize[1];
                        size_t localWorkSize[1];
                        for (int rowouter = 0; rowouter < N; rowouter++){
                            print_time();
                            std::cout << "rowouter : " << rowouter << std::endl;
                            std::unordered_set<int> rowouter_cols(Lap_ind_V.begin() + Lap_rowptr_V[rowouter],
                                                              Lap_ind_V.begin() + Lap_rowptr_V[rowouter + 1]);
                            
                            for (int r = rowouter + 1; r < N; ++r) {
                                //printVector(Lap_rowptr_V);
                                std::unordered_set<int> current_row_cols(Lap_ind_V.begin() + Lap_rowptr_V[r],
                                                                    Lap_ind_V.begin() + Lap_rowptr_V[r + 1]);
                                std::vector<int> missing_cols;
                                bool begin_check = false;
                                bool skip = false;
                                for (int col : rowouter_cols) {
                                    if (current_row_cols.find(col) == current_row_cols.end()) {
                                        if(col <= rowouter){
                                            skip = true;
                                            break;
                                        }
                                        missing_cols.push_back(col);
                                        if(!begin_check && col == rowouter){
                                            begin_check = true;
                                        }
                                    }
                                }
                                if(!skip){
                                    int insert_pos = 0;
                                    // Insert missing columns into current row
                                    if(!begin_check){
                                        insert_pos = Lap_rowptr_V[r + 1];
                                    }
                                    else{
                                        // Sort to preserve order if needed
                                        std::sort(missing_cols.begin(), missing_cols.end());
                                        insert_pos = Lap_rowptr_V[r];
                                    }
                                    
                                    for (int col : missing_cols) {
                                        Lap_ind_V.insert(Lap_ind_V.begin() + insert_pos, col);
                                        Lap_val_V.insert(Lap_val_V.begin() + insert_pos, 0.0);
                                        ++insert_pos;
                                    }
                                    // Update row pointers
                                    for (int rowp = r + 1; rowp <= N; ++rowp) {
                                        Lap_rowptr_V[rowp] += missing_cols.size();
                                    }  
                                }
                                //std::cout << skip << r << std::endl;
                                //printVector(Lap_rowptr_V);
                            }
                            clFinish(queue);
                            // std::cout << Lap_val_V.size() << ", " << Lap_ind_V.size() << std::endl;
                            err = clEnqueueWriteBuffer(queue, LFvalues.buffer, CL_FALSE, 0, sizeof(float) * Lap_ind_V.size(), Lap_val_V.data(), 0, NULL, &event1);
                            err = clEnqueueWriteBuffer(queue, Lap_ind.buffer, CL_FALSE, 0, sizeof(int) * Lap_ind_V.size(), Lap_ind_V.data(), 0, NULL, &event2);
                            err = clEnqueueWriteBuffer(queue, Lap_rowptr.buffer, CL_FALSE, 0, sizeof(int) * (N + 1), Lap_rowptr_V.data(), 0, NULL, &event3);
                            if (err != CL_SUCCESS) {
                                std::cerr << "Error writing to LFvalues buffer: " << err << std::endl;
                            }
                            clWaitForEvents(3, (cl_event[]){event1, event2, event3});
                            std::cout << Lap_rowptr_V[N] << std::endl;
                            size_t nnz = (size_t)Lap_rowptr_V[N];
                            size_t local = 64; // or whatever max workgroup size your device supports
                            if (nnz % local != 0) {
                                globalWorkSize[0] = ((nnz + local - 1) / local) * local;
                            } else {
                                globalWorkSize[0] = nnz;
                            }
                            localWorkSize[0] = local;
                            // std::cout << "after Write buffer : ";
                            // printCL(LFvalues.buffer, N * N, 1);
                            // std::cout << "Launching kernel" << std::endl;
                            err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_int), &rowouter);
                            err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
                            clFinish(queue);
                            // std::cout << "Kernel finish" << std::endl;
                            // Lap_val_V = copyCL<float>(queue, LFvalues.buffer, N * N, &event3);
                            Lap_val_V = copyCL_offset<float>(queue, LFvalues.buffer, Lap_val_V, 
                                                            Lap_rowptr_V[rowouter], 
                                                            N * N - Lap_rowptr_V[rowouter], &event3);
                            // std::cout << "Copy finish" << std::endl;
                            for (auto it = Lap_rowptr_V.rbegin(); it != Lap_rowptr_V.rend() - 1; ++it) {
                                size_t idx = *it;
                                // Safety check
                                if (idx < Lap_ind_V.size() && Lap_ind_V[idx] == 0) {
                                    Lap_val_V.erase(Lap_val_V.begin() + idx);
                                    Lap_ind_V.erase(Lap_ind_V.begin() + idx);
                                    for (size_t j = 0; j < Lap_rowptr_V.size(); ++j) {
                                        if (Lap_rowptr_V[j] > idx) {
                                            Lap_rowptr_V[j]--;
                                        }
                                    }
                                }
                            }
                        }
                        print_time();
                        std::cout << "loop end" << std::endl;
                        printVector(Lap_val_V);
                        printVector(Lap_ind_V);
                        printVector(Lap_rowptr_V);
                        csr_to_dense_and_print(Lap_rowptr_V, Lap_ind_V, Lap_val_V, N);
                        clReleaseMemObject(LFvalues.buffer);
                        clReleaseMemObject(Lap_ind.buffer);
                        clReleaseMemObject(Lap_rowptr.buffer);
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