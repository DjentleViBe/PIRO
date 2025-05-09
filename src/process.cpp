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
#include <openclutilities.hpp>

extern char* dt;
extern std::time_t now;

using namespace Piro;

int process::matchscalartovar(std::string var){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    for(int v = 0; v < MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.size(); v++){
        if(var == MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[v].Scalars){
            return v;
        }
    }
    return 0;
}
int process::matchconstanttovar(std::string var){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    
    for(int v = 0; v < MP.getvalue<std::vector<int>>(Piro::MeshParams::CONSTANTSLIST).size(); v++){
        if(var == MP.getvalue<std::vector<std::string>>(Piro::MeshParams::CONSTANTSLIST)[v]){
            return v;
        }
    }
    return 0;
}

CLBuffer process::ddt_r(std::string var){
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    int ind = process::matchscalartovar(var);
    return CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[ind];
}

const float process::ddc_r(std::string var){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    int ind = process::matchscalartovar(var);
    
    return MP.getvalue<std::vector<float>>(Piro::MeshParams::CONSTANTSVALUES)[ind];
}

CLBuffer process::r(std::string var){
    cl_int err;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
    int N = n[0] * n[1] * n[2];
    int ind = process::matchscalartovar(var);
    CLBuffer memC;

    std::vector<float> prop = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[ind].values;
    size_t globalWorkSizemultiplyconst[1] = { (size_t)N };
    memC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * N, prop.data(), &err);
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    float timestep = SP.getvalue<float>(Piro::SolveParams::TIMESTEP);
    err |= clSetKernelArg(kernels::kernel_math[6], 0, sizeof(cl_mem), &memC.buffer);
    err |= clSetKernelArg(kernels::kernel_math[6], 1, sizeof(cl_float), &timestep);
    err |= clSetKernelArg(kernels::kernel_math[6], 2, sizeof(cl_mem), &CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[ind].buffer);
    err |= clSetKernelArg(kernels::kernel_math[6], 3, sizeof(cl_uint), &N);

    err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel_math[6], 1, NULL, globalWorkSizemultiplyconst, NULL, 0, NULL, NULL);
    
    return memC;
}

CLBuffer process::laplacian_full(std::string var){
    cl_int err;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    std::vector<float> delta = SP.getvalue<std::vector<float>>(Piro::SolveParams::DELTA);
    float timestep = SP.getvalue<float>(Piro::SolveParams::TIMESTEP);

    int N = n[0] * n[1] * n[2];
    int ind = process::matchscalartovar(var);
    CLBuffer memC;
    
    /*
    std::vector<float> hostValues(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[ind].values.size());
    err = clEnqueueReadBuffer(queue, CDGPU.values_gpu[ind].buffer, CL_TRUE, 0,
                            sizeof(float) * N, hostValues.data(),
                            0, NULL, NULL);
    for (size_t i = 0; i < hostValues.size(); ++i) {
        std::cout << hostValues[i] << " ";
    }*/
    std::cout << std::endl;
    if(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[ind].type == 0){
        std::vector<float> prop = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[ind].values;
        size_t globalWorkSizelaplacian[1] = { (size_t)N };
        memC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * N, prop.data(), &err);
    
        err |= clSetKernelArg(kernels::kernel[5], 0, sizeof(cl_mem), &CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[ind].buffer);
        err |= clSetKernelArg(kernels::kernel[5], 1, sizeof(cl_mem), &memC.buffer);
        err |= clSetKernelArg(kernels::kernel[5], 2, sizeof(cl_float), &delta[0]);
        err |= clSetKernelArg(kernels::kernel[5], 3, sizeof(cl_float), &delta[1]);
        err |= clSetKernelArg(kernels::kernel[5], 4, sizeof(cl_float), &delta[2]);
        err |= clSetKernelArg(kernels::kernel[5], 5, sizeof(cl_uint), &n[0]);
        err |= clSetKernelArg(kernels::kernel[5], 6, sizeof(cl_uint), &n[1]);
        err |= clSetKernelArg(kernels::kernel[5], 7, sizeof(cl_float), &timestep);
        err |= clSetKernelArg(kernels::kernel[5], 8, sizeof(cl_uint), &N);

        err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel[5], 1, NULL, globalWorkSizelaplacian, NULL, 0, NULL, NULL);
    
    }
    else{
        std::vector<float> prop = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[ind].values;
        size_t globalWorkSizelaplacian[1] = { (size_t)3 * N };
        memC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * 3 * N, prop.data(), &err);
    
        err |= clSetKernelArg(kernels::kernel[9], 0, sizeof(cl_mem), &CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[ind].buffer);
        err |= clSetKernelArg(kernels::kernel[9], 1, sizeof(cl_mem), &memC.buffer);
        err |= clSetKernelArg(kernels::kernel[9], 2, sizeof(cl_float), &delta[0]);
        err |= clSetKernelArg(kernels::kernel[9], 3, sizeof(cl_float), &delta[1]);
        err |= clSetKernelArg(kernels::kernel[9], 4, sizeof(cl_float), &delta[2]);
        err |= clSetKernelArg(kernels::kernel[9], 5, sizeof(cl_uint), &n[0]);
        err |= clSetKernelArg(kernels::kernel[9], 6, sizeof(cl_uint), &n[1]);
        err |= clSetKernelArg(kernels::kernel[9], 7, sizeof(cl_float), &timestep);
        err |= clSetKernelArg(kernels::kernel[9], 8, sizeof(cl_uint), &N);

        err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel[9], 1, NULL, globalWorkSizelaplacian, NULL, 0, NULL, NULL);
    

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
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    if(LAP_INIT == false){
        laplacian_CSR_init(); // needs to be done just once until CDGPU.indices and CDGPU.values are filled in
        /*
        if (MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[ind].type == 0){
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
        return {CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_CSR)[0], 
                CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_CSR)[1], 
                CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_CSR)[2]};
    }
    else{
        return {CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_CSR)[0], 
                CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_CSR)[1], 
                CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_CSR)[2]};
    }
}

CLBuffer process::div_r(std::string var1, std::string var2){
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
    std::vector<float> delta = SP.getvalue<std::vector<float>>(Piro::SolveParams::DELTA);
    float timestep = SP.getvalue<float>(Piro::SolveParams::TIMESTEP);
    int ind1 = process::matchscalartovar(var1);
    int ind2 = process::matchscalartovar(var2);
    CLBuffer memC, multi;
    int N = n[0] * n[1] * n[2];
    std::vector<float> prop = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[ind1].values;
    
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
        multi.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * 3 * N, MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[ind2].values.data(), &err);

        memC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * N, prop.data(), &err);
    
        err |= clSetKernelArg(kernels::kernel[3], 0, sizeof(cl_mem), &CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[ind1].buffer);
        err |= clSetKernelArg(kernels::kernel[3], 1, sizeof(cl_mem), &memC.buffer);
        err |= clSetKernelArg(kernels::kernel[3], 2, sizeof(cl_mem), &multi.buffer);
        err |= clSetKernelArg(kernels::kernel[3], 3, sizeof(cl_float), &delta[0]);
        err |= clSetKernelArg(kernels::kernel[3], 4, sizeof(cl_float), &delta[1]);
        err |= clSetKernelArg(kernels::kernel[3], 5, sizeof(cl_float), &delta[2]);
        err |= clSetKernelArg(kernels::kernel[3], 6, sizeof(cl_uint), &n[0]);
        err |= clSetKernelArg(kernels::kernel[3], 7, sizeof(cl_uint), &n[1]);
        err |= clSetKernelArg(kernels::kernel[3], 8, sizeof(cl_float), &timestep);
        err |= clSetKernelArg(kernels::kernel[3], 9, sizeof(cl_uint), &N);

        err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel[3], 1, NULL, globalWorkSizegradient, NULL, 0, NULL, NULL);
        
    }
    else{
        // var1 = vector, var2 = vector
        size_t globalWorkSizegradient[1] = { (size_t)3 * N };
        multi.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * 3 * N, MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[ind2].values.data(), &err);

        memC.buffer = clCreateBuffer(kernels::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * 3 * N, prop.data(), &err);
    
        err |= clSetKernelArg(kernels::kernel[4], 0, sizeof(cl_mem), &CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[ind1].buffer);
        err |= clSetKernelArg(kernels::kernel[4], 1, sizeof(cl_mem), &memC.buffer);
        err |= clSetKernelArg(kernels::kernel[4], 2, sizeof(cl_mem), &multi.buffer);
        err |= clSetKernelArg(kernels::kernel[4], 3, sizeof(cl_float), &delta[0]);
        err |= clSetKernelArg(kernels::kernel[4], 4, sizeof(cl_float), &delta[1]);
        err |= clSetKernelArg(kernels::kernel[4], 5, sizeof(cl_float), &delta[2]);
        err |= clSetKernelArg(kernels::kernel[4], 6, sizeof(cl_uint), &n[0]);
        err |= clSetKernelArg(kernels::kernel[4], 7, sizeof(cl_uint), &n[1]);
        err |= clSetKernelArg(kernels::kernel[4], 8, sizeof(cl_float), &timestep);
        err |= clSetKernelArg(kernels::kernel[4], 9, sizeof(cl_uint), &N);

        err = clEnqueueNDRangeKernel(kernels::queue, kernels::kernel[4], 1, NULL, globalWorkSizegradient, NULL, 0, NULL, NULL);
        
        
    }

    return memC;
    
}
scalarMatrix::scalarMatrix(CLBuffer SM) : smatrix(SM) {
    // No assignment operator is used here
}

void scalarMatrix::Solve(float currenttime){
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    float timestep = SP.getvalue<float>(Piro::SolveParams::TIMESTEP);
    int totaltimesteps = SP.getvalue<int>(Piro::SolveParams::TOTALTIMESTEPS);
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    
    int N = n[0] * n[1] * n[2];
    ts = int(currenttime / timestep);
    Piro::logger::info("Timestep : ", ts + 1, " / ", totaltimesteps);
    // apply Boundary Conditions
    err = clEnqueueCopyBuffer(kernels::queue, smatrix.buffer, CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[0].buffer, 0, 0, sizeof(float) * N, 0, NULL, NULL);
    Piro::bc::opencl_setBC(0);
    
    if((ts + 1) % SP.getvalue<int>(Piro::SolveParams::SAVE) == 0){
        Piro::logger::info("Post processing started");
        err = clEnqueueReadBuffer(kernels::queue, CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[0].buffer, CL_TRUE, 0,
                sizeof(float) * N, MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[0].values.data(), 0, NULL, NULL);
        
        Piro::post::export_paraview(ts);
        Piro::logger::info("Post processing finished\n");
    }
}