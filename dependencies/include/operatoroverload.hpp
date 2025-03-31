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
                        // look_up = 0, 1, 2, 4 ; 1, 0, 3, 5 = 
                        // int nnz = Lap_val_V.size();
                        std::vector<float> Value_filtered_V = {0, 0, 0, 0, 0, 0, 0, 0};
                        std::vector<float> Value_filtered_E = {0, 0, 0, 0, 0, 0, 0, 0};
                        CLBuffer LFvalues, Lap_ind, Value_filtered, Value_filtered_0, Value_filtered_row, Lap_rowptr, pivot;
                        print_time();
                        std::cout << "Buffer creation begin" << std::endl;
                        
                        // int row = 0;
                        // float sizefactor = 2 / 3 ;
                        LFvalues.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(float) * N * N * 3 / 4, nullptr, &err);
                        Lap_ind.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(int) * N * N * 3 / 4, nullptr, &err);
                        Lap_rowptr.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(int) * (N + 1), nullptr, &err);
                        
                        Value_filtered.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * N, nullptr, &err);
                        Value_filtered_0.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            sizeof(float) * N, nullptr, &err);
                        Value_filtered_row.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(float) * N, nullptr, &err);
                        
                        pivot.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
                        print_time();
                        std::cout << "Buffer creation end" << std::endl;
                    
                        err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &Lap_rowptr.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &Lap_ind.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_mem), &LFvalues.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_mem), &Value_filtered.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 5, sizeof(cl_int), &N);
                        err |= clSetKernelArg(kernelfilterarray, 6, sizeof(cl_mem), &pivot.buffer);
                        
                        err |= clSetKernelArg(kernelfilterrow, 0, sizeof(cl_mem), &Value_filtered_0.buffer);
                        err |= clSetKernelArg(kernelfilterrow, 1, sizeof(cl_mem), &Value_filtered_row.buffer);
                        err |= clSetKernelArg(kernelfilterrow, 2, sizeof(cl_mem), &Value_filtered.buffer);
                        err |= clSetKernelArg(kernelfilterrow, 3, sizeof(cl_int), &N);
                        err |= clSetKernelArg(kernelfilterrow, 4, sizeof(cl_mem), &pivot.buffer);

                        err |= clSetKernelArg(kernel_math[1], 0, sizeof(cl_mem), &Value_filtered.buffer);
                        err |= clSetKernelArg(kernel_math[1], 1, sizeof(cl_mem), &Value_filtered.buffer);
                        err |= clSetKernelArg(kernel_math[1], 2, sizeof(cl_mem), &Value_filtered_row.buffer);
                        err |= clSetKernelArg(kernel_math[1], 3, sizeof(cl_int), &N);
                        
                        float fillValue = 0.0f;
                        int fillValue_int = 0;
                        
                        print_time();
                        std::cout << "Loop begin" << std::endl;
                        for (int rowouter = 0; rowouter < 1; rowouter++){
                            // std::cout << "Values size : " << Lap_val_V.size() << std::endl;
                            for (int row = rowouter; row < 1; row ++){
                                // size_t globalWorkSize_square[2] = { (size_t)N, (size_t)N};
                                size_t globalWorkSize_square[1] = { (size_t)N};
                                
                                clEnqueueFillBuffer(queue, LFvalues.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N * N * 3 / 4, 0, nullptr, nullptr);
                                clEnqueueFillBuffer(queue, Lap_ind.buffer, &fillValue_int, sizeof(int), 0, sizeof(int) * N * N * 3 / 4, 0, nullptr, nullptr);
                                clEnqueueFillBuffer(queue, Lap_rowptr.buffer, &fillValue_int, sizeof(int), 0, sizeof(int) * (N + 1), 0, nullptr, nullptr);
                                clFinish(queue);
                                
                                err = clEnqueueWriteBuffer(queue, LFvalues.buffer, CL_TRUE, 0, sizeof(float) * Lap_val_V.size(), Lap_val_V.data(), 0, NULL, NULL);
                                if (err != CL_SUCCESS) {
                                    std::cerr << "Error creating LFvalues buffer: " << err << std::endl;
                                }
                                err = clEnqueueWriteBuffer(queue, Lap_ind.buffer, CL_TRUE, 0, sizeof(int) * Lap_ind_V.size(), Lap_ind_V.data(), 0, NULL, NULL);
                                err = clEnqueueWriteBuffer(queue, Lap_rowptr.buffer, CL_TRUE, 0, sizeof(int) * (N + 1), Lap_rowptr_V.data(), 0, NULL, NULL);
                                clFinish(queue);
                                // std::cout << "iteration: " << row << std::endl;
                                err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_int), &row);
                                // step 1 : Extract the row
                                if(row == rowouter){
                                    print_time();
                                    std::cout << "start" << std::endl;
                                    clEnqueueFillBuffer(queue, Value_filtered_0.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N, 0, nullptr, nullptr);
                                    clFinish(queue);
                                    err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_mem), &Value_filtered_0.buffer);
                                    err |= clSetKernelArg(kernelfilterarray, 7, sizeof(cl_int), &rowouter);
                                    err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize_square, NULL, 0, NULL, NULL);
                                    clFinish(queue);
                                    print_time();
                                    std::cout << "end" << std::endl;
                                    printCL(pivot.buffer, 1, 1);
                                    printCL(Value_filtered_0.buffer, N, 1);
                                }
                                else{
                                    /*
                                    // print_time();
                                    // std::cout << "GPU start" << std::endl;
                                    err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_mem), &Value_filtered.buffer);
                                    clEnqueueFillBuffer(queue, Value_filtered.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N, 0, nullptr, nullptr);
                                    clEnqueueFillBuffer(queue, Value_filtered_row.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N, 0, nullptr, nullptr);
                                    clFinish(queue);
                                    err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
                                    clFinish(queue);
                                    Value_filtered_V = copyCL(Value_filtered.buffer, N, 1);
                                    // printCL(pivot.buffer, 1, 1);
                                    // step 2 : Multiply with factor = ele / pivot
                                    err |= clSetKernelArg(kernelfilterrow, 5, sizeof(cl_mem), &rowouter);
                                    err = clEnqueueNDRangeKernel(queue, kernelfilterrow, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
                                    clFinish(queue);
                                    // step 2 : subtract the row from 0th
                                    err = clEnqueueNDRangeKernel(queue, kernel_math[1], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
                                    clFinish(queue);
                                    Value_filtered_E = copyCL(Value_filtered.buffer, N, 1);
                                    // print_time();
                                    // std::cout << "GPU finish" << std::endl;
                                    // Update CSR array
                                    for (int vf = 0; vf < N; vf++){
                                        if(Value_filtered_V[vf] == 0 && Value_filtered_E[vf] != 0)
                                        {
                                            // if 0 -> value, add this element
                                            // int start = Lap_rowptr_V[row];
                                            int end = Lap_rowptr_V[row + 1];
                                            
                                            Lap_ind_V.insert(Lap_ind_V.begin() + end, vf);
                                            Lap_val_V.insert(Lap_val_V.begin() + end, Value_filtered_E[vf]);
                                            
                                            for(int rowptr = row; rowptr < N + 1; rowptr++){
                                                Lap_rowptr_V[rowptr + 1] += 1;
                                            }
                                        }
                                        // i value to value, change this element
                                        else if(Value_filtered_V[vf] != 0 && Value_filtered_E[vf] == 0)
                                        {
                                            // if value -> 0, remove this element
                                            int start = Lap_rowptr_V[row];
                                            int end = Lap_rowptr_V[row + 1];
                                            for(int col = start; col < end; col ++){
                                                if(Lap_ind_V[col] == vf){
                                                    Lap_val_V.erase(Lap_val_V.begin() + col);
                                                    Lap_ind_V.erase(Lap_ind_V.begin() + col);
                                                    break;
                                                }
                                            }
                                            
                                            for(int rowptr = row; rowptr < N + 1; rowptr++){
                                                Lap_rowptr_V[rowptr + 1] -= 1;
                                                
                                            }
                                        }
                                        else if (Value_filtered_V[vf] == 0 && Value_filtered_E[vf] == 0){
                                            continue;
                                        }
                                        else{
                                            // if value changes, edit this element
                                            // row_ptr range
                                            int start = Lap_rowptr_V[row];
                                            int end = Lap_rowptr_V[row + 1];
                                            for(int col = start; col < end; col ++){
                                                if(Lap_ind_V[col] == vf){
                                                    Lap_val_V[col] = Value_filtered_E[vf];
                                                    break;
                                                }
                                            }
                                        }
                                    }*/
                                }
                                
                                clFinish(queue);
                                
                            }
                            // std::cout << "\n";
                        }
                        print_time();
                        std::cout << "loop end" << std::endl;
                        clReleaseMemObject(LFvalues.buffer);
                        clReleaseMemObject(Lap_ind.buffer);
                        clReleaseMemObject(Lap_rowptr.buffer);
                        clReleaseMemObject(Value_filtered.buffer);
                        clReleaseMemObject(Value_filtered_0.buffer);
                        clReleaseMemObject(Value_filtered_row.buffer);
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