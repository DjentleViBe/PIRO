#include <operatordivergenceCSR.hpp>
#include <openclutilities.hpp>
#include <datatypes.hpp>
#include <stencils.hpp>
#include <preprocess.hpp>

int Piro::CSR::div(){
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
    auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) +  2];
    cd.rowpointers.assign(N + 1, 0);

    Piro::CLBuffer CD_GPU;
    std::vector<CLBuffer> div_collect_x;
    div_collect_x.push_back(CD_GPU);
    div_collect_x.push_back(CD_GPU);
    div_collect_x.push_back(CD_GPU);
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::CSR::stencils::div_cd_x(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 2){
        Piro::CSR::stencils::div_dw_x(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 1){
        Piro::CSR::stencils::div_uw_x(cd, n, norm);
    }
    
    
    div_collect_x[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) * cd.rowpointers.size(), 
                                                    cd.rowpointers.data(), &err);

    div_collect_x[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) *  cd.columns.size(), 
                                                    cd.columns.data(), &err);

    div_collect_x[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(float) *  cd.values.size(), 
                                                    cd.values.data(), &err);
    cd.columns.clear();
    cd.values.clear();
    std::vector<CLBuffer> div_collect_y;
    div_collect_y.push_back(CD_GPU);
    div_collect_y.push_back(CD_GPU);
    div_collect_y.push_back(CD_GPU);
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::CSR::stencils::div_cd_y(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 2){
        Piro::CSR::stencils::div_dw_y(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 1){
        Piro::CSR::stencils::div_uw_y(cd, n, norm);
    }
    div_collect_y[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) * cd.rowpointers.size(), 
                                                    cd.rowpointers.data(), &err);

    div_collect_y[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) *  cd.columns.size(), 
                                                    cd.columns.data(), &err);

    div_collect_y[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(float) *  cd.values.size(), 
                                                    cd.values.data(), &err);
    cd.columns.clear();
    cd.values.clear();
    std::vector<CLBuffer> div_collect_z;
    div_collect_z.push_back(CD_GPU);
    div_collect_z.push_back(CD_GPU);
    div_collect_z.push_back(CD_GPU);

    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::CSR::stencils::div_cd_z(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 2){
        Piro::CSR::stencils::div_dw_z(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 1){
        Piro::CSR::stencils::div_uw_z(cd, n, norm);
    }
    div_collect_z[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) * cd.rowpointers.size(), 
                                                    cd.rowpointers.data(), &err);

    div_collect_z[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) *  cd.columns.size(), 
                                                    cd.columns.data(), &err);

    div_collect_z[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(float) *  cd.values.size(), 
                                                    cd.values.data(), &err);
    std::vector<Piro::CLBuffer> div_collect;
    div_collect.reserve(div_collect_x.size() + div_collect_y.size() + div_collect_z.size());
    div_collect.insert(div_collect.end(), div_collect_x.begin(), div_collect_x.end());
    div_collect.insert(div_collect.end(), div_collect_y.begin(), div_collect_y.end());
    div_collect.insert(div_collect.end(), div_collect_z.begin(), div_collect_z.end());
    CDGPU.setvalue(Piro::CellDataGPU::DIV_CSR, div_collect);
    Piro::INIT::getInstance().DIV_INIT = true;
    return 0;
}