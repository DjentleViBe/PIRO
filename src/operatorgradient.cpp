#include <operatorgradient.hpp>
#include <CL/opencl.h>
#include <operatoroverload.hpp>
#include <datatypes.hpp>
#include <mathoperations.hpp>
#include <openclutilities.hpp>
#include <preprocess.hpp>
#include <bc.hpp>
#include <logger.hpp>
#include <methods.hpp>
#include <stencils.hpp>

int Piro::CSR::gradient(){
    // lets do just X direction
    cl_int err;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    int N = n[0] * n[1] * n[2];
    Piro::CellData CD;
    // Piro::logger::info(N);
    float norm = 2 * (l[0] / float(n[0] - 2));
    auto& cd = MP.getvalue<std::vector<AMR_LEVELS>>(Piro::MeshParams::AMRLEVELS)[0].amr[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + MP.getvalue<int>(Piro::MeshParams::CONSTANTNUM) +  1];
    cd.rowpointers.assign(N + 1, 0);

    Piro::CLBuffer CD_GPU;
    std::vector<CLBuffer> gradient_collect_x;
    gradient_collect_x.push_back(CD_GPU);
    gradient_collect_x.push_back(CD_GPU);
    gradient_collect_x.push_back(CD_GPU);
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::CSR::stencils::gradient_cd_x(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 2){
        Piro::CSR::stencils::gradient_dw_x(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 1){
        Piro::CSR::stencils::gradient_uw_x(cd, n, norm);
    }
    
    
    gradient_collect_x[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) * cd.rowpointers.size(), 
                                                    cd.rowpointers.data(), &err);

    gradient_collect_x[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) *  cd.columns.size(), 
                                                    cd.columns.data(), &err);

    gradient_collect_x[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(float) *  cd.values.size(), 
                                                    cd.values.data(), &err);
    cd.columns.clear();
    cd.values.clear();
    std::vector<CLBuffer> gradient_collect_y;
    gradient_collect_y.push_back(CD_GPU);
    gradient_collect_y.push_back(CD_GPU);
    gradient_collect_y.push_back(CD_GPU);
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::CSR::stencils::gradient_cd_y(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 2){
        Piro::CSR::stencils::gradient_dw_y(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 1){
        Piro::CSR::stencils::gradient_uw_y(cd, n, norm);
    }
    gradient_collect_y[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) * cd.rowpointers.size(), 
                                                    cd.rowpointers.data(), &err);

    gradient_collect_y[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) *  cd.columns.size(), 
                                                    cd.columns.data(), &err);

    gradient_collect_y[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(float) *  cd.values.size(), 
                                                    cd.values.data(), &err);
    cd.columns.clear();
    cd.values.clear();
    std::vector<CLBuffer> gradient_collect_z;
    gradient_collect_z.push_back(CD_GPU);
    gradient_collect_z.push_back(CD_GPU);
    gradient_collect_z.push_back(CD_GPU);

    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::CSR::stencils::gradient_cd_z(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 2){
        Piro::CSR::stencils::gradient_dw_z(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 1){
        Piro::CSR::stencils::gradient_uw_z(cd, n, norm);
    }
    gradient_collect_z[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) * cd.rowpointers.size(), 
                                                    cd.rowpointers.data(), &err);

    gradient_collect_z[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) *  cd.columns.size(), 
                                                    cd.columns.data(), &err);

    gradient_collect_z[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(float) *  cd.values.size(), 
                                                    cd.values.data(), &err);
    std::vector<Piro::CLBuffer> gradient_collect;
    gradient_collect.reserve(gradient_collect_x.size() + gradient_collect_y.size() + gradient_collect_z.size());
    gradient_collect.insert(gradient_collect.end(), gradient_collect_x.begin(), gradient_collect_x.end());
    gradient_collect.insert(gradient_collect.end(), gradient_collect_y.begin(), gradient_collect_y.end());
    gradient_collect.insert(gradient_collect.end(), gradient_collect_z.begin(), gradient_collect_z.end());
    CDGPU.setvalue(Piro::CellDataGPU::GRADIENT_CSR, gradient_collect);
    Piro::INIT::getInstance().GRAD_INIT = true;
    return 0;
}