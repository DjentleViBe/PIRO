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
                        cl_event event1, event2, event3, event4;
                        // cl_event event4;
                        LFvalues.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * N * N, nullptr, &err);
                        Lap_ind.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * N * N, nullptr, &err);
                        Lap_rowptr.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * (N + 1), nullptr, &err);
                        err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &Lap_rowptr.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &Lap_ind.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_mem), &LFvalues.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_int), &N);
                        
                        printVector(Lap_ind_V);
                        print_time();
                        std::cout << "Loop begin" << std::endl;
                        //std::cout << "before Write buffer : ";
                        //printVector(Lap_val_V);
                        size_t globalWorkSize[1];
                        size_t localWorkSize[1];
                        
                        for (int rowouter = 0; rowouter < N - 1; rowouter++){
                            print_time();
                            std::cout << "rowouter : " << rowouter << std::endl;
                            
                            std::unordered_set<int> rowouter_cols(Lap_ind_V.begin() + Lap_rowptr_V[rowouter],
                                                            Lap_ind_V.begin() + Lap_rowptr_V[rowouter + 1]);
                            // printVector(Lap_rowptr_V);
                            // printVector(Lap_ind_V);
                            
                            for (int r = rowouter + 1; r < N; ++r) {
                                // if(rowouter >= N - 1) printVector(Lap_rowptr_V);
                                // if(rowouter >= N - 1) printVector(Lap_ind_V);
                                std::unordered_set<int> current_row_cols(Lap_ind_V.begin() + Lap_rowptr_V[r],
                                                                    Lap_ind_V.begin() + Lap_rowptr_V[r + 1]);

                                int num_inserted = 0;
                                int row_start = Lap_rowptr_V[r];
                                int row_end   = Lap_rowptr_V[r + 1];
                                
                                for (int col : rowouter_cols) {
                                    if (current_row_cols.find(col) == current_row_cols.end()) {
                                        auto row_begin = Lap_ind_V.begin() + row_start;
                                        auto row_finish = Lap_ind_V.begin() + row_end + num_inserted;

                                        auto insert_iter = std::upper_bound(row_begin, row_finish, col);
                                        int insert_pos = insert_iter - Lap_ind_V.begin();

                                        Lap_ind_V.insert(insert_iter, col);
                                        Lap_val_V.insert(Lap_val_V.begin() + insert_pos, 0.0);

                                        ++num_inserted;

                                        // if(rowouter >= N - 2) std::cout << "col : " << col << "insert_pos : " << insert_pos << " " << ", row_end : " << row_end << "\n";
                                    }
                                }
                                if (num_inserted != 0){
                                    for (int i = r + 1; i < Lap_rowptr_V.size(); ++i) {
                                        Lap_rowptr_V[i] += num_inserted;
                                        // if(rowouter >= N - 2) std::cout << rowouter << std::endl;
                                    }
                                }
                            }
                            clFinish(queue);
                            int* rowptr_ptr = (int*)clEnqueueMapBuffer(queue, Lap_rowptr.buffer, CL_TRUE, CL_MAP_WRITE, sizeof(int) * rowouter, sizeof(int) * (N + 1 - rowouter), 0, nullptr, nullptr, &err);
                            int* ind_ptr = (int*)clEnqueueMapBuffer(queue, Lap_ind.buffer , CL_TRUE, CL_MAP_WRITE, sizeof(int) * Lap_rowptr_V[rowouter], sizeof(int) * (N * N - Lap_rowptr_V[rowouter]), 0, nullptr, nullptr, &err);
                            float* values_ptr = (float*)clEnqueueMapBuffer(queue, LFvalues.buffer, CL_TRUE, CL_MAP_WRITE, sizeof(float) * Lap_rowptr_V[rowouter], sizeof(float) * (N * N - Lap_rowptr_V[rowouter]), 0, nullptr, nullptr, &err);
                        
                            // std::cout << Lap_val_V.size() << ", " << Lap_ind_V.size() << std::endl;
                            std::memcpy(rowptr_ptr, Lap_rowptr_V.data() + rowouter, sizeof(int) *  N + 1 - rowouter);
                            std::memcpy(ind_ptr, Lap_ind_V.data() + Lap_rowptr_V[rowouter], sizeof(int) * (Lap_ind_V.size() - Lap_rowptr_V[rowouter]));
                            std::memcpy(values_ptr, Lap_val_V.data() + Lap_rowptr_V[rowouter], sizeof(float) *  (Lap_val_V.size() - Lap_rowptr_V[rowouter]));

                            err = clEnqueueUnmapMemObject(queue, Lap_rowptr.buffer, rowptr_ptr, 0, nullptr, &event1);
                            err = clEnqueueUnmapMemObject(queue, Lap_ind.buffer, ind_ptr, 0, nullptr, &event2);
                            err = clEnqueueUnmapMemObject(queue, LFvalues.buffer, values_ptr, 0, nullptr, &event3);
                            clWaitForEvents(3, (cl_event[]){event1, event2, event3});
                         
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
                            err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_int), &rowouter);
                            err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
                            clFinish(queue);

                            Lap_val_V = copyCL_offset<float>(queue, LFvalues.buffer, Lap_val_V, 
                                                            Lap_rowptr_V[rowouter], 
                                                            N * N - Lap_rowptr_V[rowouter], &event4);
                            
                            // std::cout << "Copy finish" << std::endl;
                            std::vector<int> new_rowptr;
                            std::vector<int> new_colind;
                            std::vector<float> new_values;
                            new_rowptr.push_back(0);
    
                            for (int i = 0; i < N; ++i) {
                                int row_start = Lap_rowptr_V[i];
                                int row_end = Lap_rowptr_V[i + 1];
                                
                                for (int j = row_start; j < row_end; ++j) {
                                    if (std::abs(Lap_val_V[j]) > 1E-6) {
                                        new_colind.push_back(Lap_ind_V[j]);
                                        new_values.push_back(Lap_val_V[j]);
                                    }
                                }
                                new_rowptr.push_back(static_cast<int>(new_values.size()));
                            }

                            Lap_rowptr_V = std::move(new_rowptr);
                            Lap_ind_V = std::move(new_colind);
                            Lap_val_V = std::move(new_values);
                            // csr_to_dense_and_print(Lap_rowptr_V, Lap_ind_V, Lap_val_V, N);
                        }
                        print_time();
                        std::cout << "loop end" << std::endl;
                        // change the last element
                        printVector(Lap_val_V);
                        printVector(Lap_ind_V);
                        printVector(Lap_rowptr_V);
                        csr_to_dense_and_print(Lap_rowptr_V, Lap_ind_V, Lap_val_V, N);
                        clReleaseMemObject(LFvalues.buffer);
                        clReleaseMemObject(Lap_ind.buffer);
                        clReleaseMemObject(Lap_rowptr.buffer);
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