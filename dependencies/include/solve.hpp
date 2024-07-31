#ifndef solve_hpp
#define solve_hpp

#include "datatypes.hpp"
#include "preprocess.hpp"
#include "bc.hpp"
#include "extras.hpp"
#include "postprocess.hpp"
#include "operatoroverload.hpp"
#include <iostream>
#include <immintrin.h>
#include <ctime>
#include "gpuinit.hpp"
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#elif _WIN32
    #include "./CL/opencl.h"
#else
    #include "./CL/opencl.h"
#endif
extern Giro::SolveParams SP;
extern char* dt;
extern std::time_t now;

int solve();
namespace Giro{
    class MathOperations{
        public:
            int idx(int i, int j, int k, int N_x, int N_y) {
                return i + j * N_x + k * N_x * N_y;
            }
    };

    class Solve{
        private:
            // bool sgm = false;
        public:
            int matchscalartovar(std::string var){
                    for(int v = 0; v < MP.AMR[0].CD.size(); v++){
                        if(var == MP.AMR[0].CD[v].Scalars){
                            return v;
                        }
                    }
                    return 0;
                }
            int matchconstanttovar(std::string var){
                    for(int v = 0; v < MP.constantslist.size(); v++){
                        if(var == MP.constantslist[v]){
                            return v;
                        }
                    }
                    return 0;
                }

            CLBuffer ddt_r(std::string var){
                int ind = matchscalartovar(var);
                
                return CDGPU.values_gpu[ind];
            }

            const float ddc_r(std::string var){
                int ind = matchscalartovar(var);
                
                return MP.constantsvalues[ind];
            }

            CLBuffer laplacian_r(std::string var){
                cl_int err;
                int N = MP.n[0] * MP.n[1] * MP.n[2];
                int ind = matchscalartovar(var);
                // std::cout << "ind : " << ind << std::endl;
                std::vector<float> prop = MP.AMR[0].CD[ind].values;
                size_t globalWorkSizelaplacian[1] = { (size_t)N };
                // call openkernel for laplacian
                // CLBuffer memB;
                CLBuffer memC;
                // memB.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                //          sizeof(float) * N, MP.AMR[0].CD[ind].values.data(), &err);
                memC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N, prop.data(), &err);
                
                err |= clSetKernelArg(kernellaplacian, 0, sizeof(cl_mem), &CDGPU.values_gpu[ind].buffer);
                err |= clSetKernelArg(kernellaplacian, 1, sizeof(cl_mem), &memC.buffer);
                err |= clSetKernelArg(kernellaplacian, 2, sizeof(cl_float), &SP.delta[0]);
                err |= clSetKernelArg(kernellaplacian, 3, sizeof(cl_float), &SP.delta[1]);
                err |= clSetKernelArg(kernellaplacian, 4, sizeof(cl_float), &SP.delta[2]);
                err |= clSetKernelArg(kernellaplacian, 5, sizeof(cl_uint), &MP.n[0]);
                err |= clSetKernelArg(kernellaplacian, 6, sizeof(cl_uint), &MP.n[1]);
                err |= clSetKernelArg(kernellaplacian, 7, sizeof(cl_float), &SP.timestep);
                err |= clSetKernelArg(kernellaplacian, 8, sizeof(cl_uint), &N);

                err = clEnqueueNDRangeKernel(queue, kernellaplacian, 1, NULL, globalWorkSizelaplacian, NULL, 0, NULL, NULL);
                
                //err = clEnqueueReadBuffer(queue, memC.buffer, CL_TRUE, 0,
                //              sizeof(float) * N, prop.data(), 0, NULL, NULL);
                //std::cout << "laplacian" << std::endl;
                //printVector(prop);

                return memC;
            }

            CLBuffer div_r(std::string var1, std::string var2){
                CLBuffer memC, multi;
                int N = MP.n[0] * MP.n[1] * MP.n[2];
                int ind1 = matchscalartovar(var1);
              
                // vector itself
                multi.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * 3 * N, MP.AMR[0].CD[matchscalartovar(var2)].values.data(), &err);
                
                std::vector<float> prop = MP.AMR[0].CD[ind1].values;
                size_t globalWorkSizegradient[1] = { (size_t)N };
                
                
                // memB.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                //          sizeof(float) * N, MP.AMR[0].CD[ind].values.data(), &err);
                memC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, prop.data(), &err);
                
                err |= clSetKernelArg(kernelgradient, 0, sizeof(cl_mem), &CDGPU.values_gpu[ind1].buffer);
                err |= clSetKernelArg(kernelgradient, 1, sizeof(cl_mem), &memC.buffer);
                err |= clSetKernelArg(kernelgradient, 2, sizeof(cl_mem), &multi.buffer);
                err |= clSetKernelArg(kernelgradient, 3, sizeof(cl_float), &SP.delta[0]);
                err |= clSetKernelArg(kernelgradient, 4, sizeof(cl_float), &SP.delta[1]);
                err |= clSetKernelArg(kernelgradient, 5, sizeof(cl_float), &SP.delta[2]);
                err |= clSetKernelArg(kernelgradient, 6, sizeof(cl_uint), &MP.n[0]);
                err |= clSetKernelArg(kernelgradient, 7, sizeof(cl_uint), &MP.n[1]);
                err |= clSetKernelArg(kernelgradient, 8, sizeof(cl_float), &SP.timestep);
                err |= clSetKernelArg(kernelgradient, 9, sizeof(cl_uint), &N);

                err = clEnqueueNDRangeKernel(queue, kernelgradient, 1, NULL, globalWorkSizegradient, NULL, 0, NULL, NULL);
                
                //err = clEnqueueReadBuffer(queue, memC.buffer, CL_TRUE, 0,
                //              sizeof(float) * N, prop.data(), 0, NULL, NULL);
                //std::cout << "laplacian" << std::endl;
                //printVector(prop);

                return memC;
                
            }

        };

        class scalarMatrix{
            private:
                CLBuffer smatrix;
            public:
                scalarMatrix(CLBuffer SM){
                    smatrix = SM;
                }

                void Solve(float currenttime){
                    int N = MP.n[0] * MP.n[1] * MP.n[2];
                    ts = int(currenttime / SP.timestep);
                    std::cout << "Timestep : " << ts + 1  << " / " << SP.totaltimesteps << std::endl;
                    // apply Boundary Conditions
                    err = clEnqueueCopyBuffer(queue, smatrix.buffer, CDGPU.values_gpu[0].buffer, 0, 0, sizeof(float) * N, 0, NULL, NULL);
                    // std::cout << "timestep" << std::endl;
                    // err = clEnqueueReadBuffer(queue, CDGPU.values_gpu[0].buffer, CL_TRUE, 0,
                    //          sizeof(float) * N, MP.AMR[0].CD[0].values.data(), 0, NULL, NULL);
                    // printVector(MP.AMR[0].CD[0].values);
                    opencl_setBC(0);
                    
                    // export every timestep
                    // printVector(MP.AMR[0].CD[0].values);
                    //std::cout << "after solving" << std::endl;
                    //printVector(MP.AMR[0].CD[0].values);
                    if((ts + 1) % SP.save == 0){
                        std::cout << "Post processing started" << std::endl;
                        err = clEnqueueReadBuffer(queue, CDGPU.values_gpu[0].buffer, CL_TRUE, 0,
                              sizeof(float) * N, MP.AMR[0].CD[0].values.data(), 0, NULL, NULL);
                        print_time();
                        postprocess("T");
                        print_time();
                        std::cout << "Post processing finished" << std::endl;
                    }
                }
        };
        
};

#endif