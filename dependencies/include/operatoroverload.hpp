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
                        /*
                        int nnz = MP.AMR[0].CD[MP.vectornum + MP.scalarnum].values.size();
                        std::vector<float>A(nnz, 0.0);
                        std::vector<int>A_int(nnz, 0);

                        
                        std::vector<CLBuffer> L(3);
                        
                        L[0].buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * (N + 1), A_int.data(), &err);
                        L[1].buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * nnz, A_int.data(), &err);
                        L[2].buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * nnz, A.data(), &err);
                        
                        std::vector<CLBuffer> U(3);
                        U[0].buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * (N + 1), A_int.data(), &err);
                        U[1].buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * nnz, A_int.data(), &err);
                        U[2].buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * nnz, A_int.data(), &err);

                        // CLBuffer b, y, x;
                        // b.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        //    sizeof(float) * N, A.data(), &err);
                        
                        // take the inverse of the RHS
                        std::cout << "LU Decomposition kernel" << std::endl;
                        err |= clSetKernelArg(kernellu_decomposition, 0, sizeof(cl_mem), &other[0].buffer);
                        err |= clSetKernelArg(kernellu_decomposition, 1, sizeof(cl_mem), &other[1].buffer);
                        err |= clSetKernelArg(kernellu_decomposition, 2, sizeof(cl_mem), &other[2].buffer);

                        err |= clSetKernelArg(kernellu_decomposition, 3, sizeof(cl_mem), &L[0].buffer);
                        err |= clSetKernelArg(kernellu_decomposition, 4, sizeof(cl_mem), &L[1].buffer);
                        err |= clSetKernelArg(kernellu_decomposition, 5, sizeof(cl_mem), &L[2].buffer);

                        err |= clSetKernelArg(kernellu_decomposition, 6, sizeof(cl_mem), &U[0].buffer);
                        err |= clSetKernelArg(kernellu_decomposition, 7, sizeof(cl_mem), &U[1].buffer);
                        err |= clSetKernelArg(kernellu_decomposition, 8, sizeof(cl_mem), &U[2].buffer);
                        err |= clSetKernelArg(kernellu_decomposition, 9, sizeof(cl_int), &N);
                        err = clEnqueueNDRangeKernel(queue, kernellu_decomposition, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
                        clFinish(queue);

                        err |= clSetKernelArg(kernelforward_substitution_csr, 0, sizeof(cl_mem), &L[0].buffer);
                        err |= clSetKernelArg(kernelforward_substitution_csr, 1, sizeof(cl_mem), &L[1].buffer);
                        err |= clSetKernelArg(kernelforward_substitution_csr, 2, sizeof(cl_mem), &L[2].buffer);
                        err |= clSetKernelArg(kernelforward_substitution_csr, 3, sizeof(cl_mem), &partC.buffer);
                        err |= clSetKernelArg(kernelforward_substitution_csr, 4, sizeof(cl_mem), &this->buffer);
                        err |= clSetKernelArg(kernelforward_substitution_csr, 5, sizeof(cl_int), &N);
                        err = clEnqueueNDRangeKernel(queue, kernelforward_substitution_csr, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
                        clFinish(queue);

                        err |= clSetKernelArg(kernelbackward_substitution_csr, 0, sizeof(cl_mem), &U[0].buffer);
                        err |= clSetKernelArg(kernelbackward_substitution_csr, 1, sizeof(cl_mem), &U[1].buffer);
                        err |= clSetKernelArg(kernelbackward_substitution_csr, 2, sizeof(cl_mem), &U[2].buffer);
                        err |= clSetKernelArg(kernelbackward_substitution_csr, 3, sizeof(cl_mem), &partD.buffer);
                        err |= clSetKernelArg(kernelbackward_substitution_csr, 4, sizeof(cl_mem), &partC.buffer);
                        err |= clSetKernelArg(kernelbackward_substitution_csr, 5, sizeof(cl_int), &N);
                        err = clEnqueueNDRangeKernel(queue, kernelbackward_substitution_csr, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
                        clFinish(queue);*/
                        size_t globalWorkSize_square[1] = { (size_t)N};
                        std::vector<float> Lap_full = {-6.0,1.0,1.0,0.0,1.0,0.0,0.0,0.0,
                                                        1.0,-6.0,0.0,1.0,0.0,1.0,0.0,0.0,
                                                        1.0,0.0,-6.0,1.0,0.0,0.0,1.0,0.0,
                                                        0.0,1.0,1.0,-6.0,0.0,0.0,0.0,1.0,
                                                        1.0,0.0,0.0,0.0,-6.0,1.0,1.0,0.0,
                                                        0.0,1.0,0.0,0.0,1.0,-6.0,0.0,1.0,
                                                        0.0,0.0,1.0,0.0,1.0,0.0,-6.0,1.0,
                                                        0.0,0.0,0.0,1.0,0.0,1.0,1.0,-6.0,};
                        int N = 8;
                        
                        std::vector<float> Lap_val_V = {-6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1, -6, 1, 1, 1};
                        std::vector<int> Lap_ind_V = {0, 1, 2, 4, 1, 0, 3, 5, 2, 3, 0, 6, 3, 2, 1, 7, 4, 5, 6, 0, 5, 4, 7, 1, 6, 7, 4, 2, 7, 6, 5, 3};
                        std::vector<int> Lap_rowptr_V = {0, 4, 8, 12, 16, 20, 24, 28, 32};
                        int nnz = Lap_val_V.size();
                        std::vector<float> Value_filtered_V = {0, 0, 0, 0, 0, 0, 0, 0};
                        CLBuffer LF, LFvalues, Lap_ind, Value_filtered, Lap_rowptr, LTM;
                        std::vector<float>LT_matrix(N * N, 0.0);
                        LF.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * N * N, Lap_full.data(), &err);
                        LFvalues.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * nnz, Lap_val_V.data(), &err);
                        Lap_ind.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * N * N, Lap_ind_V.data(), &err);
                        Lap_rowptr.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * N * N, Lap_rowptr_V.data(), &err);
                        LTM.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * N * N, LT_matrix.data(), &err);
                        //err |= clSetKernelArg(kernellu_decompose_dense, 0, sizeof(cl_mem), &LF.buffer);
                        //err |= clSetKernelArg(kernellu_decompose_dense, 1, sizeof(cl_mem), &LTM.buffer);
                        //err |= clSetKernelArg(kernellu_decompose_dense, 2, sizeof(cl_int), &N);
                        //err = clEnqueueNDRangeKernel(queue, kernellu_decompose_dense, 2, NULL, globalWorkSize_square, NULL, 0, NULL, NULL);
                        //clFinish(queue);
                        
                        int row = 1;
                        LFvalues.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * nnz, Lap_val_V.data(), &err);
                        Lap_ind.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * nnz, Lap_ind_V.data(), &err);

                        Lap_rowptr.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * N, Lap_rowptr_V.data(), &err);
                        Value_filtered.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * N, Value_filtered_V.data(), &err);
                        
                        err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &Lap_rowptr.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &Lap_ind.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_mem), &LFvalues.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_mem), &Value_filtered.buffer);
                        err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_int), &row);
                        err |= clSetKernelArg(kernelfilterarray, 5, sizeof(cl_int), &N);
                        err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize_square, NULL, 0, NULL, NULL);
                        clFinish(queue);
                        std::cout << "LU Decomposition print" << std::endl;

                        
                        printCL(Value_filtered.buffer, N, 1);
                        // printCL(LFind.buffer, N*N, 0);
                        // printCL(LFrowptr.buffer, N*N, 0);
                        // printCLArray(LF.buffer, N, 1);
                        std::cout << "LU Decomposition finish" << std::endl;
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