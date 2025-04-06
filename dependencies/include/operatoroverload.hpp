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
                // other[0] = rowpointers
                // other[1] = columns
                // other[2] = values
                std::cout << "Backward Euler" << std::endl;
                    if(SP.solverscheme == 17){
                        std::cout << "LU Decomposition" << std::endl;
                        cl_event event1, event2, event3, event4, event5, event6, event7, event8, event9;
                        int N = MP.n[0] * MP.n[1] * MP.n[2];
                        int factor = N * N * 3 / 4;
                        std::cout << N << std::endl;
                        std::cout << factor << std::endl;
                        std::cout << N * N << std::endl;
                        size_t globalWorkSize_square[1] = { (size_t)N};
                        if(RHS_INIT == false){
                            float fillValue = 0.0f;
                            int fillValue_int = 0;
                            int index = MP.vectornum + MP.scalarnum;
                            // std::cout << N << std::endl;
                            RHS.operandrowptr = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(int) * (N + 1), nullptr, &err);
                            RHS.operandcolumns = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(int) * factor, nullptr, &err);
                            RHS.operandvalues = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(float) * factor, nullptr, &err);

                            clEnqueueFillBuffer(queue, RHS.operandvalues, &fillValue, sizeof(float), 0, sizeof(float) *factor, 0, nullptr, nullptr);
                            clEnqueueFillBuffer(queue, RHS.operandcolumns, &fillValue_int, sizeof(int), 0, sizeof(int) *factor, 0, nullptr, nullptr);
                            clEnqueueFillBuffer(queue, RHS.operandrowptr, &fillValue_int, sizeof(int), 0, sizeof(int) * (N + 1), 0, nullptr, nullptr);
                            clFinish(queue);
                            clEnqueueCopyBuffer(queue, other[2].buffer, RHS.operandvalues, 0, 0, sizeof(float) * RHS.sparsecount, 0, NULL, NULL);
                            clEnqueueCopyBuffer(queue, other[1].buffer, RHS.operandcolumns, 0, 0, sizeof(int) * RHS.sparsecount, 0, NULL, NULL);
                            clEnqueueCopyBuffer(queue, other[0].buffer, RHS.operandrowptr, 0, 0, sizeof(int) * (N + 1), 0, NULL, NULL);
                            clFinish(queue);
                            MP.AMR[0].CD[index].values = copyCL<float>(queue, other[2].buffer, RHS.sparsecount);
                            MP.AMR[0].CD[index].columns = copyCL<int>(queue, other[1].buffer, RHS.sparsecount);
                            MP.AMR[0].CD[index].rowpointers = copyCL<int>(queue, other[0].buffer, N + 1);
                            std::vector<float> Value_filtered_V(N);
                            std::vector<float> Value_filtered_E(N);
                            clFinish(queue);
                            // printVector(MP.AMR[0].CD[index].values);
                            // printVector(MP.AMR[0].CD[index].columns);
                            // printVector(MP.AMR[0].CD[index].rowpointers);
                            // std::cout << "for python" << std::endl;
                            CLBuffer Value_filtered, Value_filtered_0, Value_filtered_row, pivot;
                            Value_filtered.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, nullptr, &err);
                            Value_filtered_0.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, nullptr, &err);
                            Value_filtered_row.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, nullptr, &err);
                            pivot.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
                            // create a local copy of values, ind and rowptr of the RHS
                            
                            err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &RHS.operandrowptr);
                            err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &RHS.operandcolumns);
                            err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_mem), &RHS.operandvalues);
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
                            print_time();
                            std::cout << "loop begin" << std::endl;
                            for (int rowouter = 0; rowouter < N; rowouter++){
                                err |= clSetKernelArg(kernelfilterrow, 5, sizeof(cl_mem), &rowouter);
                                // std::cout << "Values size : " << Lap_val_V.size() << std::endl;
                                for (int row = rowouter; row < N; row ++){
                                    clEnqueueFillBuffer(queue, RHS.operandvalues, &fillValue, sizeof(float), 0, sizeof(float) * factor, 0, nullptr, &event3);
                                    clEnqueueFillBuffer(queue, RHS.operandcolumns, &fillValue_int, sizeof(int), 0, sizeof(int) * factor, 0, nullptr, &event4);
                                    clEnqueueFillBuffer(queue, RHS.operandrowptr, &fillValue_int, sizeof(int), 0, sizeof(int) * (N + 1), 0, nullptr, &event5);
                                    clWaitForEvents(3, (cl_event[]){event3, event4, event5});
                                    err = clEnqueueWriteBuffer(queue, RHS.operandvalues, CL_TRUE, 0, sizeof(float) * MP.AMR[0].CD[index].values.size(), MP.AMR[0].CD[index].values.data(), 0, NULL, &event6);
                                    if (err != CL_SUCCESS) { printf("Error writing operandvalues: %d\n", err); }
                                    err = clEnqueueWriteBuffer(queue, RHS.operandcolumns, CL_TRUE, 0, sizeof(int) * MP.AMR[0].CD[index].columns.size(), MP.AMR[0].CD[index].columns.data(), 0, NULL, &event7);
                                    if (err != CL_SUCCESS) { printf("Error writing operandcolumns: %d\n", err); }
                                    err = clEnqueueWriteBuffer(queue, RHS.operandrowptr, CL_TRUE, 0, sizeof(int) * (N + 1), MP.AMR[0].CD[index].rowpointers.data(), 0, NULL, &event8);
                                    if (err != CL_SUCCESS) { printf("Error writing operandrowptr: %d\n", err); }
                                    clWaitForEvents(3, (cl_event[]){event6, event7, event8});
                                    //std::cout << "inputArrayvalues after write: ";
                                    //printCL(RHS.operandvalues, factor, 1);
                                    err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_int), &row);
                                    // step 1 : Extract the row
                                    if(row == rowouter){
                                        clEnqueueFillBuffer(queue, Value_filtered_0.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N, 0, nullptr, &event9);
                                        clWaitForEvents(1, (cl_event[]){event9});
                                        err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_mem), &Value_filtered_0.buffer);
                                        err |= clSetKernelArg(kernelfilterarray, 7, sizeof(cl_int), &rowouter);
                                        err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize_square, NULL, 0, NULL, NULL);
                                        clFinish(queue);
                                        // std::cout << row << " array : \n";
                                    }
                                    else{
                                        err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_mem), &Value_filtered.buffer);
                                        size_t buffer_size = sizeof(float) * N;

                                        err = clEnqueueFillBuffer(queue, Value_filtered.buffer, &fillValue, sizeof(float), 0, buffer_size, 0, nullptr, &event1);
                                        if (err != CL_SUCCESS) { printf("Error filling Value_filtered : %d\n", err); }
                                        clEnqueueFillBuffer(queue, Value_filtered_row.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N, 0, nullptr, &event2);
                                        clWaitForEvents(2, (cl_event[]){event1, event2});
                                
                                        // std::cout << "Value_filtered_buffer after fill: ";
                                        // printCL(Value_filtered.buffer, N, 1);
                                        //std::cout << "inputArrayrow: ";
                                        //printCL(RHS.operandrowptr, N + 1, 0);
                                        //std::cout << "inputArraycol: ";
                                        //printCL(RHS.operandcolumns, factor, 0);
                                        //std::cout << "inputArrayvalues: ";
                                        //printCL(RHS.operandvalues, factor, 1);
                                        err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize_square, NULL, 0, NULL, NULL);
                                        clFinish(queue);
                                        // std::cout << "Value_filtered_buffer: ";
                                        // printCL(Value_filtered.buffer, N, 1);
                                        Value_filtered_V = copyCL<float>(queue, Value_filtered.buffer, N);
                                        // std::cout << "kernelfilterarray: ";
                                        // printVector(Value_filtered_V);
                                        // step 2 : Multiply with factor = ele / pivot
                                        err = clEnqueueNDRangeKernel(queue, kernelfilterrow, 1, NULL, globalWorkSize_square, NULL, 0, NULL, NULL);
                                        clFinish(queue);
                                        // std::cout << "kernelfilterrow: ";
                                        //printCL(Value_filtered_row.buffer, N, 1);
                                        err = clEnqueueNDRangeKernel(queue, kernel_math[1], 1, NULL, globalWorkSize_square, NULL, 0, NULL, NULL);
                                        clFinish(queue);
                                        Value_filtered_E = copyCL<float>(queue, Value_filtered.buffer, N);
                                        // clFinish(queue);
                                        // printCL(pivot.buffer, 1, 1);
                                        // printVector(Value_filtered_E);
                                        
                                        // printVector(MP.AMR[0].CD[index].values);
                                        //printCL(RHS.operandvalues, factor, 1);
                                        for (int vf = 0; vf < N; vf++){
                                            if(std::abs(Value_filtered_V[vf]) < EPSILON && std::abs(Value_filtered_E[vf]) > EPSILON)
                                            {
                                                // if 0 -> value, add this element
                                                // int start = Lap_rowptr_V[row];
                                                int end = MP.AMR[0].CD[index].rowpointers[row + 1];
                                                
                                                MP.AMR[0].CD[index].columns.insert(MP.AMR[0].CD[index].columns.begin() + end, vf);
                                                MP.AMR[0].CD[index].values.insert(MP.AMR[0].CD[index].values.begin() + end, Value_filtered_E[vf]);
                                                
                                                for(int rowptr = row + 1; rowptr <= N; rowptr++){
                                                    MP.AMR[0].CD[index].rowpointers[rowptr] += 1;
                                                }
                                            }
                                            // i value to value, change this element
                                            else if(std::abs(Value_filtered_V[vf]) > EPSILON && std::abs(Value_filtered_E[vf]) < EPSILON)
                                            {
                                                // if value -> 0, remove this element
                                                int start = MP.AMR[0].CD[index].rowpointers[row];
                                                int end = MP.AMR[0].CD[index].rowpointers[row + 1];
                                                for(int col = start; col < end; col ++){
                                                    if(MP.AMR[0].CD[index].columns[col] == vf){
                                                        MP.AMR[0].CD[index].values.erase(MP.AMR[0].CD[index].values.begin() + col);
                                                        MP.AMR[0].CD[index].columns.erase(MP.AMR[0].CD[index].columns.begin() + col);
                                                        break;
                                                    }
                                                }
                                                
                                                for(int rowptr = row + 1; rowptr <= N; rowptr++){
                                                    MP.AMR[0].CD[index].rowpointers[rowptr] -= 1;
                                                    
                                                }
                                            }
                                            else if (std::abs(Value_filtered_V[vf]) < EPSILON && std::abs(Value_filtered_E[vf]) < EPSILON){
                                                continue;
                                            }
                                            else if(std::abs(Value_filtered_V[vf]) != std::abs(Value_filtered_E[vf])){
                                                // if value changes, edit this element
                                                // row_ptr range
                                                int start = MP.AMR[0].CD[index].rowpointers[row];
                                                int end = MP.AMR[0].CD[index].rowpointers[row + 1];
                                                for(int col = start; col < end; col ++){
                                                    if(MP.AMR[0].CD[index].columns[col] == vf){
                                                        MP.AMR[0].CD[index].values[col] = Value_filtered_E[vf];
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    clFinish(queue);
                                }
                            }
                            print_time();
                            std::cout << "loop end" << std::endl;
                            clEnqueueFillBuffer(queue, Value_filtered.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N, 0, nullptr, nullptr);
                            clEnqueueFillBuffer(queue, Value_filtered_0.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N, 0, nullptr, nullptr);
                            clEnqueueFillBuffer(queue, Value_filtered_row.buffer, &fillValue, sizeof(float), 0, sizeof(float) * N, 0, nullptr, nullptr);
                              
                            clReleaseMemObject(Value_filtered.buffer);
                            clReleaseMemObject(Value_filtered_0.buffer);
                            clReleaseMemObject(Value_filtered_row.buffer);
                            clReleaseMemObject(pivot.buffer);
                            std::cout << MP.AMR[0].CD[index].values.size() << std::endl;
                            // printVector(MP.AMR[0].CD[index].values);
                            //printCL(RHS.operandvalues, factor, 1);
                            //printCL(RHS.operandrowptr, N + 1, 0);
                            RHS_INIT = true;
                        }
                        else{
                            printCL(RHS.operandvalues,factor, 1);
                        }
                        
                        
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