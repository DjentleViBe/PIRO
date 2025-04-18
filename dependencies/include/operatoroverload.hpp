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
                        std::cout << "LU Decomposition" << std::endl;
                        int N = MP.n[0] * MP.n[1] * MP.n[2];
                        int index = MP.vectornum + MP.scalarnum;
                        print_time();
                        std::cout << "RHS_INIT begin, sparse count : " << RHS.sparsecount << std::endl;
                        if(RHS_INIT == false){
                            cl_event event0, event1, event2, event3, event4, event7, event8, event9;
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
                            RHS.operandrows = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * N * N, nullptr, &err);
                            
                            clEnqueueFillBuffer(queue, RHS.operandvalues, &fillValue, sizeof(float), 0, sizeof(float) * N * N, 0, nullptr, nullptr);
                            clEnqueueFillBuffer(queue, RHS.operandcolumns, &fillValue_int, sizeof(int), 0, sizeof(int) * N * N, 0, nullptr, nullptr);
                            clEnqueueFillBuffer(queue, RHS.operandrowptr, &fillValue_int, sizeof(int), 0, sizeof(int) * (N + 1), 0, nullptr, nullptr);
                            clEnqueueFillBuffer(queue, RHS.operandrows, &fillValue_int, sizeof(int), 0, sizeof(int) * N * N, 0, nullptr, nullptr);
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
                            MP.AMR[0].CD[index].rows.resize(MP.AMR[0].CD[index].columns.size());
                            for (int row = 0; row < MP.AMR[0].CD[index].rowpointers.size() - 1; ++row) {
                                for (int i = MP.AMR[0].CD[index].rowpointers[row]; i < MP.AMR[0].CD[index].rowpointers[row + 1]; ++i) {
                                    MP.AMR[0].CD[index].rows[i] = row;
                                }
                            }   
                            /////////////////////////////////////////////////////////////////////////////////////////
                            
                            err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &RHS.operandrowptr);
                            err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &RHS.operandrows);
                            err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_mem), &RHS.operandcolumns);
                            err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_mem), &RHS.operandvalues);
                            err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_int), &N);

                            size_t globalWorkSize[1];
                            size_t localWorkSize[1];
                            // csr_to_dense_and_print(MP.AMR[0].CD[index].rowpointers, MP.AMR[0].CD[index].columns, MP.AMR[0].CD[index].values, N);
                            auto& cd = MP.AMR[0].CD[index];
                            print_time();
                            std::cout << "Loop begin" << std::endl;
                            std::vector<int> new_rowptr(N + 1);
                            std::vector<int> new_colind(N * N);
                            std::vector<int> new_rowind(N * N);
                            std::vector<float> new_values(N * N);
                            for (int rowouter = 0; rowouter < 1; rowouter++){
                                print_time();
                                std::cout << "rowouter : " << rowouter << std::endl;
                                auto columns_begin = cd.columns.begin();
                                std::vector<int> rowouter_cols_sorted(columns_begin + cd.rowpointers[rowouter],
                                                                    columns_begin + cd.rowpointers[rowouter + 1]);
                                std::sort(rowouter_cols_sorted.begin(), rowouter_cols_sorted.end());
                                for (int r = rowouter + 1; r < N; ++r) {
                                    //printVector(cd.rowpointers);
                                    std::vector<int> current_row_sorted(columns_begin + cd.rowpointers[r],
                                                                            columns_begin + cd.rowpointers[r + 1]);
                                    std::sort(current_row_sorted.begin(), current_row_sorted.end());
                                    std::vector<int> missing_cols;
                                    
                                    missing_cols.reserve(rowouter_cols_sorted.size());
                                    std::set_difference(rowouter_cols_sorted.begin(), rowouter_cols_sorted.end(),
                                                        current_row_sorted.begin(), current_row_sorted.end(),
                                                        std::back_inserter(missing_cols));
                                
                                    int insert_pos = 0;
                                    insert_pos = cd.rowpointers[r];
                                    
                                    for (int col : missing_cols) {
                                        cd.columns.insert(columns_begin + insert_pos, col);
                                        cd.rows.insert(cd.rows.begin() + insert_pos, r);
                                        cd.values.insert(cd.values.begin() + insert_pos, 0.0);
                                        ++insert_pos;
                                    }
                                    // Update row pointers
                                    for (int rowp = r + 1; rowp <= N; ++rowp) {
                                        cd.rowpointers[rowp] += missing_cols.size();
                                    }  
                                    
                                    //std::cout << skip << r << std::endl;
                                    //printVector(cd.rowpointers);
                                }                           
                                print_time();
                                std::cout << "Inserting 0s finished\n";
                                err = clEnqueueWriteBuffer(queue, RHS.operandrowptr, CL_FALSE, 
                                                            sizeof(int) * rowouter, 
                                                            sizeof(int) * (N + 1 - rowouter),
                                                            cd.rowpointers.data() + rowouter, 
                                                            0, nullptr, &event0);
          
                                err = clEnqueueWriteBuffer(queue, RHS.operandcolumns, CL_FALSE, 
                                                            sizeof(int) * cd.rowpointers[rowouter], 
                                                            sizeof(int) * (cd.columns.size() - cd.rowpointers[rowouter]),
                                                            cd.columns.data() + cd.rowpointers[rowouter], 
                                                            0, nullptr, &event1);
                                
                                err = clEnqueueWriteBuffer(queue, RHS.operandrows, CL_FALSE, 
                                                            sizeof(int) * cd.rowpointers[rowouter], 
                                                            sizeof(int) * (cd.rows.size() - cd.rowpointers[rowouter]),
                                                            cd.rows.data() + cd.rowpointers[rowouter], 
                                                            0, nullptr, &event2);
                                
                                err = clEnqueueWriteBuffer(queue, RHS.operandvalues, CL_FALSE, 
                                                            sizeof(float) * cd.rowpointers[rowouter], 
                                                            sizeof(float) * (cd.values.size() - cd.rowpointers[rowouter]),
                                                            cd.values.data() + cd.rowpointers[rowouter], 
                                                            0, nullptr, &event3);
                                
                                // Wait for all transfers to complete
                                clWaitForEvents(4, (cl_event[]){event0, event1, event2, event3});
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
                                
                                err |= clSetKernelArg(kernelfilterarray, 5, sizeof(cl_int), &rowouter);
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

                                
                                new_rowptr[0] = 0.0;
                                int index = 0;
                                for (int i = 0; i < N; ++i) {
                                    int row_start = cd.rowpointers[i];
                                    int row_end = cd.rowpointers[i + 1];
                                    
                                    for (int j = row_start; j < row_end; ++j) {
                                        if (std::abs(cd.values[j]) > 1E-6) {
                                            new_rowind[index] = cd.rows[j];
                                            new_colind[index] = cd.columns[j];
                                            new_values[index] = cd.values[j];
                                            index++;
                                        }
                                    }
                                    new_rowptr[i + 1] = index;
                                }

                                cd.rowpointers = new_rowptr;
                                cd.columns = new_colind;
                                cd.rows = new_rowind;
                                cd.values = new_values;
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