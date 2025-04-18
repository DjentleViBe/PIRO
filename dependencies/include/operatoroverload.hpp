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
                        std::vector<float> sorted_vals;
                        std::vector<int> sorted_indices;
                        std::vector<int> Lap_col_V(Lap_ind_V.size());
                        std::cout << "Loop begin" << std::endl;
                        std::vector<int> new_rowptr(N + 1);
                        std::vector<int> new_colind(N * N);
                        std::vector<int> new_rowind(N * N);
                        std::vector<float> new_values(N * N);
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
                        // populate lap_col_V
                        for (int row = 0; row < Lap_rowptr_V.size() - 1; ++row) {
                            for (int i = Lap_rowptr_V[row]; i < Lap_rowptr_V[row + 1]; ++i) {
                                Lap_col_V[i] = row;
                            }
                        }
                        /////////////////////////////////////////////////////////////////////////////////////////
                        
                        CLBuffer LFvalues, Lap_ind, Lap_rowptr, Lap_col;
                        cl_event event0, event1, event2, event3, event4;
                        // cl_event event4;
                        LFvalues.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(float) * N * N, nullptr, &err);
                        Lap_ind.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(int) * N * N, nullptr, &err);
                        Lap_col.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(int) * N * N, nullptr, &err);
                        Lap_rowptr.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(int) * (N + 1), nullptr, &err);
                        err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &Lap_rowptr.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &Lap_col.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_mem), &Lap_ind.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_mem), &LFvalues.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_int), &N);
                        
                        print_time();
                        std::cout << "Loop begin" << std::endl;
                        //std::cout << "before Write buffer : ";
                        //printVector(Lap_val_V);
                        size_t globalWorkSize[1];
                        size_t localWorkSize[1];
                        Lap_ind_V.resize(N * N , 0);
                        Lap_val_V.resize(N * N , 0.0);
                        Lap_col_V.resize(N * N , 0);
                        // printVector(Lap_ind_V);
                        for (int rowouter = 0; rowouter < N - 1; rowouter++){
                            print_time();
                            std::cout << "rowouter : " << rowouter << std::endl;
                            printVector(Lap_ind_V);
                            printVector(Lap_rowptr_V);
                            std::unordered_set<int> rowouter_cols(Lap_ind_V.begin() + Lap_rowptr_V[rowouter],
                                                            Lap_ind_V.begin() + Lap_rowptr_V[rowouter + 1]);
                            
                            for (int r = rowouter + 1; r < N; ++r) {
                                std::unordered_set<int> current_row_cols(Lap_ind_V.begin() + Lap_rowptr_V[r],
                                                                    Lap_ind_V.begin() + Lap_rowptr_V[r + 1]);
                                
                                if (current_row_cols.find(rowouter) == current_row_cols.end()) {
                                    std::cout << "skip : " << r << std::endl; 
                                    continue;
                                }
                                std::vector<int> missing_cols;
                                int num_inserted = 0;
                                for (int col : rowouter_cols) {
                                    if(col < rowouter) continue;
                                    if (current_row_cols.find(col) == current_row_cols.end()) {
                                        missing_cols.push_back(col);
                                        num_inserted++;
                                    }
                                }
                                int insert_pos = Lap_rowptr_V[r] + rowouter;
                                std::cout << "row : " << r << ", Missing cols : ";
                                printVector(missing_cols);
                                if (num_inserted != 0){
                                    std::vector<int> new_rows(num_inserted, r);
                                    std::vector<float> new_values(num_inserted, 0.0);
                                    Lap_col_V.insert(Lap_col_V.begin() + insert_pos, new_rows.begin(), new_rows.end());
                                    Lap_ind_V.insert(Lap_ind_V.begin() + insert_pos, missing_cols.begin(), missing_cols.end());
                                    Lap_val_V.insert(Lap_val_V.begin() + insert_pos, new_values.begin(), new_values.end());

                                    for (int i = r + 1; i <= Lap_rowptr_V.size(); ++i) {
                                        Lap_rowptr_V[i] += num_inserted;
                                    }
                                }
                            }
                            Lap_col_V.resize(N * N);
                            Lap_ind_V.resize(N * N);
                            Lap_val_V.resize(N * N);                
                            print_time();
                            printVector(Lap_col_V);
                            printVector(Lap_ind_V);
                            printVector(Lap_val_V);
                            printVector(Lap_rowptr_V);
                            // clFinish(queue);
                            // std::cout << "rowouter = " << rowouter << ", Lap_rowptr_V.size() = " << Lap_rowptr_V.size() << std::endl;
                            err = clEnqueueWriteBuffer(queue, Lap_rowptr.buffer, CL_FALSE, 
                                                        sizeof(int) * rowouter, 
                                                        sizeof(int) * (N + 1 - rowouter),
                                                        Lap_rowptr_V.data() + rowouter, 
                                                        0, nullptr, &event0);
                            // assert(Lap_rowptr_V.data() + rowouter != nullptr);
                            err = clEnqueueWriteBuffer(queue, Lap_ind.buffer, CL_FALSE, 
                                                        sizeof(int) * Lap_rowptr_V[rowouter], 
                                                        sizeof(int) * (N * N - Lap_rowptr_V[rowouter]),
                                                        Lap_ind_V.data() + Lap_rowptr_V[rowouter], 
                                                        0, nullptr, &event1);
                            
                            err = clEnqueueWriteBuffer(queue, Lap_col.buffer, CL_FALSE, 
                                                        sizeof(int) * Lap_rowptr_V[rowouter], 
                                                        sizeof(int) * (N * N - Lap_rowptr_V[rowouter]),
                                                        Lap_col_V.data() + Lap_rowptr_V[rowouter], 
                                                        0, nullptr, &event2);
                            
                            err = clEnqueueWriteBuffer(queue, LFvalues.buffer, CL_FALSE, 
                                                        sizeof(float) * Lap_rowptr_V[rowouter], 
                                                        sizeof(float) * (N * N - Lap_rowptr_V[rowouter]),
                                                        Lap_val_V.data() + Lap_rowptr_V[rowouter], 
                                                        0, nullptr, &event3);
                            
                            // Wait for all transfers to complete
                            clWaitForEvents(4, (cl_event[]){event0, event1, event2, event3});
                            std::cout << "write buffer lap_rowptr\n";
                            
                            size_t nnz = (size_t)Lap_rowptr_V[N];
                            size_t local = 2; // or whatever max workgroup size your device supports
                            if (nnz % local != 0) {
                                globalWorkSize[0] = ((nnz + local - 1) / local) * local;
                            } else {
                                globalWorkSize[0] = nnz;
                            }
                            localWorkSize[0] = local;
                            // std::cout << "after Write buffer : ";
                            
                            // std::cout << "Launching kernel" << std::endl;
                            err |= clSetKernelArg(kernelfilterarray, 5, sizeof(cl_int), &rowouter);
                            err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
                            clFinish(queue);
                            
                            Lap_val_V = copyCL_offset<float>(queue, LFvalues.buffer, Lap_val_V, 
                                                            Lap_rowptr_V[rowouter], 
                                                            N * N - Lap_rowptr_V[rowouter], &event4);
                            
                            // std::cout << "Copy finish" << std::endl;
                            // printVector(Lap_val_V);
                            // printVector(Lap_ind_V);

                            new_rowptr[0] = 0.0;
                            int index = 0;
                            for (int i = 0; i < N; i++) {
                                int row_start = Lap_rowptr_V[i];
                                int row_end = Lap_rowptr_V[i + 1];
                                
                                for (int j = row_start; j < row_end; j++) {
                                    if (std::abs(Lap_val_V[j]) > 1E-6) {
                                        new_rowind[index] = Lap_col_V[j];
                                        new_colind[index] = Lap_ind_V[j];
                                        new_values[index] = Lap_val_V[j];
                                        index++;
                                    }
                                }
                                new_rowptr[i + 1] = index;
                            }

                            Lap_rowptr_V = new_rowptr;
                            Lap_ind_V = new_colind;
                            Lap_col_V = new_rowind;
                            Lap_val_V = new_values; 
                            // std::cout << "" << std::endl;
                            // printVector(Lap_val_V);
                            //csr_to_dense_and_print(Lap_rowptr_V, Lap_ind_V, Lap_val_V, N);
                        }
                        print_time();
                        std::cout << "loop end" << std::endl;
                        printVector(Lap_val_V);
                        printVector(Lap_ind_V);
                        printVector(Lap_col_V);
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