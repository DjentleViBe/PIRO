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
#include <unordered_set>
#include <algorithm>

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
                        int index = MP.vectornum + MP.scalarnum;
                        print_time();
                        std::cout << "RHS_INIT begin, sparse count : " << RHS.sparsecount << std::endl;
                        if(RHS_INIT == false){
                            cl_event event1, event2, event3, event4, event7, event8, event9, event10, event11, event12;
                            print_time();
                            std::cout << "Buffer creation begin" << std::endl;
                            // calculate remaining memory
                            // std::cout << "Total variable count : " << MP.scalarnum + MP.vectornum << std::endl;
                            // std::cout << "Total memory : " << globalMemSize << std::endl;
                            // std::cout << "Max alloc : " << maxAllocSize << std::endl;

                            cl_ulong rem = globalMemSize - sizeof(float) * (2 * N + RHS.sparsecount) - sizeof(int) * (2 * N + 1 + RHS.sparsecount);
                            // std::cout << "remaining : " << rem << std::endl;
                            cl_ulong check = (sizeof(int) + sizeof(float)) * (N * N);
                            if( check < rem){
                                // std::cout << "One pass" << std::endl;
                                // check if N*N is passing for max alloc
                                if(check < maxAllocSize){
                                    // std::cout << "Buffer size within Max Allocoation size" << std::endl;
                                }
                            }
                                  
                            float fillValue = 0.0f;
                            int fillValue_int = 0;
                            RHS.operandrowptr = clCreateBuffer(context, CL_MEM_READ_WRITE  | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * (N + 1), nullptr, &err);
                            RHS.operandcolumns = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * N * N, nullptr, &err);
                            RHS.operandvalues = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * N * N, nullptr, &err);
                            clEnqueueFillBuffer(queue, RHS.operandvalues, &fillValue, sizeof(float), 0, sizeof(float) * N * N, 0, nullptr, nullptr);
                            clEnqueueFillBuffer(queue, RHS.operandcolumns, &fillValue_int, sizeof(int), 0, sizeof(int) * N * N, 0, nullptr, nullptr);
                            clEnqueueFillBuffer(queue, RHS.operandrowptr, &fillValue_int, sizeof(int), 0, sizeof(int) * (N + 1), 0, nullptr, nullptr);
                            clFinish(queue);
                            clEnqueueCopyBuffer(queue, other[2].buffer, RHS.operandvalues, 0, 0, sizeof(float) * RHS.sparsecount, 0, NULL, NULL);
                            clEnqueueCopyBuffer(queue, other[1].buffer, RHS.operandcolumns, 0, 0, sizeof(int) * RHS.sparsecount, 0, NULL, NULL);
                            clEnqueueCopyBuffer(queue, other[0].buffer, RHS.operandrowptr, 0, 0, sizeof(int) * (N + 1), 0, NULL, NULL);
                            clFinish(queue);
                            
                            MP.AMR[0].CD[index].rowpointers = copyCL<int>(queue, RHS.operandrowptr, N + 1, &event7);
                            MP.AMR[0].CD[index].columns = copyCL<int>(queue, RHS.operandcolumns, N * N, &event8);
                            MP.AMR[0].CD[index].values = copyCL<float>(queue, RHS.operandvalues, N * N, &event9);

                            print_time();
                            std::cout << "Buffer creation end" << std::endl;
                            std::vector<float> sorted_vals;
                            std::vector<int> sorted_indices;
                            
                            // Loop through each row
                            for (int row = 0; row < MP.AMR[0].CD[index].rowpointers.size() - 1; ++row) {
                                int start = MP.AMR[0].CD[index].rowpointers[row];
                                int end = MP.AMR[0].CD[index].rowpointers[row + 1];
                        
                                // Extract the elements of the current row
                                std::vector<int> row_indices(MP.AMR[0].CD[index].columns.begin() + start, MP.AMR[0].CD[index].columns.begin() + end);
                                std::vector<float> row_values(MP.AMR[0].CD[index].values.begin() + start, MP.AMR[0].CD[index].values.begin() + end);
                        
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
                            MP.AMR[0].CD[index].values = sorted_vals;
                            MP.AMR[0].CD[index].columns = sorted_indices;    
                            /////////////////////////////////////////////////////////////////////////////////////////
                            
                            err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &RHS.operandrowptr);
                            err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &RHS.operandcolumns);
                            err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_mem), &RHS.operandvalues);
                            err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_int), &N);

                            size_t globalWorkSize[1];
                            size_t localWorkSize[1];
                            // csr_to_dense_and_print(MP.AMR[0].CD[index].rowpointers, MP.AMR[0].CD[index].columns, MP.AMR[0].CD[index].values, N);
                            auto& cd = MP.AMR[0].CD[index];
                            print_time();
                            std::cout << "Loop begin" << std::endl;
                            for (int rowouter = 0; rowouter < 1; rowouter++){
                                print_time();
                                std::cout << "rowouter : " << rowouter << std::endl;
                                std::unordered_set<int> rowouter_cols(cd.columns.begin() + cd.rowpointers[rowouter],
                                                                        cd.columns.begin() + cd.rowpointers[rowouter + 1]);
                                print_time();
                                std::cout << "Inserting 0s\n";
                                
                                for (int r = rowouter + 1; r < N; ++r) {
                                    std::unordered_set<int> current_row_cols(cd.columns.begin() + cd.rowpointers[r],
                                                                    cd.columns.begin() + cd.rowpointers[r + 1]);

                                    int num_inserted = 0;
                                    int row_start = cd.rowpointers[r];
                                    int row_end   = cd.rowpointers[r + 1];
                                
                                    for (int col : rowouter_cols) {
                                        if (current_row_cols.find(col) == current_row_cols.end()) {
                                            auto row_begin = cd.columns.begin() + row_start;
                                            auto row_finish = cd.columns.begin() + row_end + num_inserted;

                                            auto insert_iter = std::upper_bound(row_begin, row_finish, col);
                                            int insert_pos = insert_iter - cd.columns.begin();

                                            cd.columns.insert(insert_iter, col);
                                            cd.values.insert(cd.values.begin() + insert_pos, 0.0);

                                            ++num_inserted;

                                            // if(rowouter >= N - 2) std::cout << "col : " << col << "insert_pos : " << insert_pos << " " << ", row_end : " << row_end << "\n";
                                        }
                                    }
                                    if (num_inserted != 0){
                                        for (int i = r + 1; i < cd.rowpointers.size(); ++i) {
                                            cd.rowpointers[i] += num_inserted;
                                            // if(rowouter >= N - 2) std::cout << rowouter << std::endl;
                                        }
                                    } 
                                }
                                print_time();
                                std::cout << "Inserting 0s finished\n";
                                int* rowptr_ptr = (int*)clEnqueueMapBuffer(queue, RHS.operandrowptr, CL_FALSE, CL_MAP_WRITE, sizeof(int) * rowouter, sizeof(int) * (N + 1 - rowouter), 0, nullptr, &event10, &err);
                                int* ind_ptr = (int*)clEnqueueMapBuffer(queue, RHS.operandcolumns, CL_FALSE, CL_MAP_WRITE, sizeof(int) * cd.rowpointers[rowouter], sizeof(int) * (N * N - cd.rowpointers[rowouter]), 0, nullptr, &event11, &err);
                                float* values_ptr = (float*)clEnqueueMapBuffer(queue, RHS.operandvalues, CL_FALSE, CL_MAP_WRITE, sizeof(float) * cd.rowpointers[rowouter], sizeof(float) * (N * N - cd.rowpointers[rowouter]), 0, nullptr, &event12, &err);
                                clWaitForEvents(3, (cl_event[]){event10, event11, event12});
                                
                                std::memcpy(rowptr_ptr, cd.rowpointers.data() + rowouter, sizeof(int) *  N + 1 - rowouter);
                                std::memcpy(ind_ptr, cd.columns.data() + cd.rowpointers[rowouter], sizeof(int) * (cd.columns.size() - cd.rowpointers[rowouter]));
                                std::memcpy(values_ptr, cd.values.data() + cd.rowpointers[rowouter], sizeof(float) *  (cd.values.size() - cd.rowpointers[rowouter]));
    
                                err = clEnqueueUnmapMemObject(queue, RHS.operandrowptr, rowptr_ptr, 0, nullptr, &event1);
                                err = clEnqueueUnmapMemObject(queue, RHS.operandcolumns, ind_ptr, 0, nullptr, &event2);
                                err = clEnqueueUnmapMemObject(queue, RHS.operandvalues, values_ptr, 0, nullptr, &event3);
                            
                                clWaitForEvents(3, (cl_event[]){event1, event2, event3});
                                print_time();
                                std::cout << "Map memory object finished\n";
                                // std::cout << cd.rowpointers[N] << std::endl;
                                size_t nnz = (size_t)cd.rowpointers[N];
                                size_t local = (size_t)maxWorkGroupSize; // or whatever max workgroup size your device supports
                                if (nnz % local != 0) {
                                    globalWorkSize[0] = ((nnz + local - 1) / local) * local;
                                } else {
                                    globalWorkSize[0] = nnz;
                                }
                                localWorkSize[0] = local;
                                // size_t localWorkSize[1] = { globalWorkSize[0] / 4 };
                                
                                err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_int), &rowouter);
                                err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
                                clFinish(queue);
                                print_time();
                                std::cout << "Kernel finished\n";
                                
                                cd.values = copyCL_offset<float>(queue, RHS.operandvalues, 
                                                                                cd.values, 
                                                                                cd.rowpointers[rowouter], 
                                                                                N * N - cd.rowpointers[rowouter], &event4);
                                print_time();
                                std::cout << "CopyCL\n";

                                // Local reference for better access
                                auto& data = cd;

                                // Calculate total non-zero elements for efficient memory allocation
                                int total_nonzeros = 0;
                                for (int i = 0; i < N; ++i) {
                                    total_nonzeros += data.rowpointers[i + 1] - data.rowpointers[i];
                                }

                                // Reserve memory up front for the vectors
                                std::vector<int> new_rowptr(N + 1);
                                std::vector<int> new_colind;
                                std::vector<float> new_values;
                                new_colind.reserve(total_nonzeros);  // Reserve memory for all non-zero entries
                                new_values.reserve(total_nonzeros);  // Same for values

                                new_rowptr[0] = 0;

                                int new_values_count = 0;
                                for (int i = 0; i < N; ++i) {
                                    int row_start = data.rowpointers[i];
                                    int row_end = data.rowpointers[i + 1];

                                    // Process non-zero elements in the row
                                    for (int j = row_start; j < row_end; ++j) {
                                        if (std::abs(data.values[j]) > 1E-6) {
                                            new_colind.push_back(data.columns[j]);
                                            new_values.push_back(data.values[j]);
                                            ++new_values_count;
                                        }
                                    }

                                    // Set the new row pointer to the count of non-zero values so far
                                    new_rowptr[i + 1] = new_values_count;
                                }

                                // Move the new vectors into the original structure
                                data.rowpointers = std::move(new_rowptr);
                                data.columns = std::move(new_colind);
                                data.values = std::move(new_values);

                                print_time();
                                std::cout << "Values erased\n";
                            }
                            print_time();
                            std::cout << "loop end" << std::endl;
                            // printVector(MP.AMR[0].CD[index].values);
                            // printVector(MP.AMR[0].CD[index].rowpointers);
                            // csr_to_dense_and_print(MP.AMR[0].CD[index].rowpointers,
                            //                         MP.AMR[0].CD[index].columns, 
                            //                         MP.AMR[0].CD[index].values, N);
                            RHS_INIT = true;
                        }
                        print_time();
                        std::cout << "RHS_INIT end" << std::endl;
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