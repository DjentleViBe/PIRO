#include <datatypes.hpp>
#include <preprocess.hpp>
#include <bc.hpp>
#include <extras.hpp>
#include <postprocess.hpp>
#include <process.hpp>
#include <operatoroverload.hpp>
#include <printutilities.hpp>
#include <iostream>
#include <immintrin.h>
#include <ctime>
#include <gpuinit.hpp>
#include <CL/opencl.h>
#include <logger.hpp>

extern Piro::SolveParams SP;
extern char* dt;
extern std::time_t now;

using namespace Piro;

int process::matchscalartovar(std::string var){
    for(int v = 0; v < MP.AMR[0].CD.size(); v++){
        if(var == MP.AMR[0].CD[v].Scalars){
            return v;
        }
    }
    return 0;
}
int process::matchconstanttovar(std::string var){
    for(int v = 0; v < MP.constantslist.size(); v++){
        if(var == MP.constantslist[v]){
            return v;
        }
    }
    return 0;
}

CLBuffer process::ddt_r(std::string var){
    int ind = process::matchscalartovar(var);
    return CDGPU.values_gpu[ind];
}

const float process::ddc_r(std::string var){
    int ind = process::matchscalartovar(var);
    
    return MP.constantsvalues[ind];
}

CLBuffer process::r(std::string var){
    cl_int err;
    int N = MP.n[0] * MP.n[1] * MP.n[2];
    int ind = process::matchscalartovar(var);
    CLBuffer memC;
    std::vector<float> prop = MP.AMR[0].CD[ind].values;
    size_t globalWorkSizemultiplyconst[1] = { (size_t)N };
    memC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * N, prop.data(), &err);
    
    err |= clSetKernelArg(kernel_math[6], 2, sizeof(cl_mem), &CDGPU.values_gpu[ind].buffer);
    err |= clSetKernelArg(kernel_math[6], 0, sizeof(cl_mem), &memC.buffer);
    err |= clSetKernelArg(kernel_math[6], 1, sizeof(cl_float), &SP.timestep);
    err |= clSetKernelArg(kernel_math[6], 3, sizeof(cl_uint), &N);

    err = clEnqueueNDRangeKernel(queue, kernel_math[6], 1, NULL, globalWorkSizemultiplyconst, NULL, 0, NULL, NULL);
    
    return memC;
}

CLBuffer process::laplacian_full(std::string var){
    cl_int err;
    int N = MP.n[0] * MP.n[1] * MP.n[2];
    int ind = process::matchscalartovar(var);
    CLBuffer memC;
    /*
    std::vector<float> hostValues(MP.AMR[0].CD[ind].values.size());
    err = clEnqueueReadBuffer(queue, CDGPU.values_gpu[ind].buffer, CL_TRUE, 0,
                            sizeof(float) * N, hostValues.data(),
                            0, NULL, NULL);
    for (size_t i = 0; i < hostValues.size(); ++i) {
        std::cout << hostValues[i] << " ";
    }*/
    std::cout << std::endl;
    if(MP.AMR[0].CD[ind].type == 0){
        std::vector<float> prop = MP.AMR[0].CD[ind].values;
        size_t globalWorkSizelaplacian[1] = { (size_t)N };
        memC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * N, prop.data(), &err);
    
        err |= clSetKernelArg(kernellaplacianscalar, 0, sizeof(cl_mem), &CDGPU.values_gpu[ind].buffer);
        err |= clSetKernelArg(kernellaplacianscalar, 1, sizeof(cl_mem), &memC.buffer);
        err |= clSetKernelArg(kernellaplacianscalar, 2, sizeof(cl_float), &SP.delta[0]);
        err |= clSetKernelArg(kernellaplacianscalar, 3, sizeof(cl_float), &SP.delta[1]);
        err |= clSetKernelArg(kernellaplacianscalar, 4, sizeof(cl_float), &SP.delta[2]);
        err |= clSetKernelArg(kernellaplacianscalar, 5, sizeof(cl_uint), &MP.n[0]);
        err |= clSetKernelArg(kernellaplacianscalar, 6, sizeof(cl_uint), &MP.n[1]);
        err |= clSetKernelArg(kernellaplacianscalar, 7, sizeof(cl_float), &SP.timestep);
        err |= clSetKernelArg(kernellaplacianscalar, 8, sizeof(cl_uint), &N);

        err = clEnqueueNDRangeKernel(queue, kernellaplacianscalar, 1, NULL, globalWorkSizelaplacian, NULL, 0, NULL, NULL);
    
    }
    else{
        std::vector<float> prop = MP.AMR[0].CD[ind].values;
        size_t globalWorkSizelaplacian[1] = { (size_t)3 * N };
        memC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * 3 * N, prop.data(), &err);
    
        err |= clSetKernelArg(kernellaplacianvector, 0, sizeof(cl_mem), &CDGPU.values_gpu[ind].buffer);
        err |= clSetKernelArg(kernellaplacianvector, 1, sizeof(cl_mem), &memC.buffer);
        err |= clSetKernelArg(kernellaplacianvector, 2, sizeof(cl_float), &SP.delta[0]);
        err |= clSetKernelArg(kernellaplacianvector, 3, sizeof(cl_float), &SP.delta[1]);
        err |= clSetKernelArg(kernellaplacianvector, 4, sizeof(cl_float), &SP.delta[2]);
        err |= clSetKernelArg(kernellaplacianvector, 5, sizeof(cl_uint), &MP.n[0]);
        err |= clSetKernelArg(kernellaplacianvector, 6, sizeof(cl_uint), &MP.n[1]);
        err |= clSetKernelArg(kernellaplacianvector, 7, sizeof(cl_float), &SP.timestep);
        err |= clSetKernelArg(kernellaplacianvector, 8, sizeof(cl_uint), &N);

        err = clEnqueueNDRangeKernel(queue, kernellaplacianvector, 1, NULL, globalWorkSizelaplacian, NULL, 0, NULL, NULL);
    

    }
    /*
    err = clEnqueueReadBuffer(queue, memC.buffer, CL_TRUE, 0,
                            sizeof(float) * N, hostValues.data(),
                            0, NULL, NULL);
    for (size_t i = 0; i < hostValues.size(); ++i) {
        std::cout << hostValues[i] << " ";
    }
    std::cout << std::endl;
    std::cout << hostValues.size() << std::endl;*/
    return memC;
    }

std::vector<CLBuffer> process::laplacian_CSR(std::string var1, std::string var2){
    // int ind = matchscalartovar(var2);
    if(LAP_INIT == false){
        laplacian_CSR_init(); // needs to be done just once until CDGPU.indices and CDGPU.values are filled in
        /*
        if (MP.AMR[0].CD[ind].type == 0){
            // scale the values
            cl_int err;
            int N = MP.n[0] * MP.n[1] * MP.n[2];
            std::vector<float>A(N, 0.0);
            CLBuffer partC;
            partC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * N, A.data(), &err);

            
            size_t globalWorkSize[1] = { (size_t)N };
            err |= clSetKernelArg(kernel_math[2], 0, sizeof(cl_mem), &partC.buffer);
            err |= clSetKernelArg(kernel_math[2], 1, sizeof(cl_mem), &CDGPU.values_gpu[ind].buffer);
            err |= clSetKernelArg(kernel_math[2], 2, sizeof(cl_mem), &CDGPU.laplacian_csr[2].buffer);
            err |= clSetKernelArg(kernel_math[2], 3, sizeof(cl_uint), &N);
            err = clEnqueueNDRangeKernel(queue, kernel_math[2], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
            */
        return {CDGPU.laplacian_csr[0], CDGPU.laplacian_csr[1], CDGPU.laplacian_csr[2]};
    }
    else{
        return {CDGPU.laplacian_csr[0], CDGPU.laplacian_csr[1], CDGPU.laplacian_csr[2]};
    }
}

CLBuffer process::div_r(std::string var1, std::string var2){
    int ind1 = process::matchscalartovar(var1);
    int ind2 = process::matchscalartovar(var2);
    CLBuffer memC, multi;
    int N = MP.n[0] * MP.n[1] * MP.n[2];
    std::vector<float> prop = MP.AMR[0].CD[ind1].values;
    
    if(ind1 == 0 && ind2 == 0){
        // std::cout << "selection" << std::endl;
        // var1 = scalar, var2 = scalar

    }
    else if(ind1 == 1 && ind2 == 0){
        // var1 = vector, var2 = scalar
        
        
    }
    else if(ind1 == 0 && ind2 == 1){
        // std::cout << "selection" << std::endl;
        // var1 = scalar, var2 = vector
        // vector itself
        size_t globalWorkSizegradient[1] = { (size_t)N };
        multi.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * 3 * N, MP.AMR[0].CD[ind2].values.data(), &err);

        memC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * N, prop.data(), &err);
    
        err |= clSetKernelArg(kernelgradient_type3, 0, sizeof(cl_mem), &CDGPU.values_gpu[ind1].buffer);
        err |= clSetKernelArg(kernelgradient_type3, 1, sizeof(cl_mem), &memC.buffer);
        err |= clSetKernelArg(kernelgradient_type3, 2, sizeof(cl_mem), &multi.buffer);
        err |= clSetKernelArg(kernelgradient_type3, 3, sizeof(cl_float), &SP.delta[0]);
        err |= clSetKernelArg(kernelgradient_type3, 4, sizeof(cl_float), &SP.delta[1]);
        err |= clSetKernelArg(kernelgradient_type3, 5, sizeof(cl_float), &SP.delta[2]);
        err |= clSetKernelArg(kernelgradient_type3, 6, sizeof(cl_uint), &MP.n[0]);
        err |= clSetKernelArg(kernelgradient_type3, 7, sizeof(cl_uint), &MP.n[1]);
        err |= clSetKernelArg(kernelgradient_type3, 8, sizeof(cl_float), &SP.timestep);
        err |= clSetKernelArg(kernelgradient_type3, 9, sizeof(cl_uint), &N);

        err = clEnqueueNDRangeKernel(queue, kernelgradient_type3, 1, NULL, globalWorkSizegradient, NULL, 0, NULL, NULL);
        
    }
    else{
        // var1 = vector, var2 = vector
        size_t globalWorkSizegradient[1] = { (size_t)3 * N };
        multi.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * 3 * N, MP.AMR[0].CD[ind2].values.data(), &err);

        memC.buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * 3 * N, prop.data(), &err);
    
        err |= clSetKernelArg(kernelgradient_type4, 0, sizeof(cl_mem), &CDGPU.values_gpu[ind1].buffer);
        err |= clSetKernelArg(kernelgradient_type4, 1, sizeof(cl_mem), &memC.buffer);
        err |= clSetKernelArg(kernelgradient_type4, 2, sizeof(cl_mem), &multi.buffer);
        err |= clSetKernelArg(kernelgradient_type4, 3, sizeof(cl_float), &SP.delta[0]);
        err |= clSetKernelArg(kernelgradient_type4, 4, sizeof(cl_float), &SP.delta[1]);
        err |= clSetKernelArg(kernelgradient_type4, 5, sizeof(cl_float), &SP.delta[2]);
        err |= clSetKernelArg(kernelgradient_type4, 6, sizeof(cl_uint), &MP.n[0]);
        err |= clSetKernelArg(kernelgradient_type4, 7, sizeof(cl_uint), &MP.n[1]);
        err |= clSetKernelArg(kernelgradient_type4, 8, sizeof(cl_float), &SP.timestep);
        err |= clSetKernelArg(kernelgradient_type4, 9, sizeof(cl_uint), &N);

        err = clEnqueueNDRangeKernel(queue, kernelgradient_type4, 1, NULL, globalWorkSizegradient, NULL, 0, NULL, NULL);
        
        
    }

    return memC;
    
}
scalarMatrix::scalarMatrix(CLBuffer SM) : smatrix(SM) {
    // No assignment operator is used here
}

void scalarMatrix::Solve(float currenttime){
    int N = MP.n[0] * MP.n[1] * MP.n[2];
    ts = int(currenttime / SP.timestep);
    Piro::Logger::info("Timestep : ", ts + 1, " / ", SP.totaltimesteps);
    // apply Boundary Conditions
    err = clEnqueueCopyBuffer(queue, smatrix.buffer, CDGPU.values_gpu[0].buffer, 0, 0, sizeof(float) * N, 0, NULL, NULL);
    Piro::bc::opencl_setBC(0);
    
    if((ts + 1) % SP.save == 0){
        Piro::Logger::info("Post processing started");
        err = clEnqueueReadBuffer(queue, CDGPU.values_gpu[0].buffer, CL_TRUE, 0,
                sizeof(float) * N, MP.AMR[0].CD[0].values.data(), 0, NULL, NULL);
        writevth(ts);
        Piro::Logger::info("Post processing finished\n");
    }
}