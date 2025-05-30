#include <operatorlaplacian.hpp>
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

int Piro::CSR::laplacian(){
    Piro::CLBuffer CD_GPU;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    int N = n[0] * n[1] * n[2];
    Piro::CellData CD;
    std::vector<CLBuffer> laplacian_collect;
    // MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
    laplacian_collect.push_back(CD_GPU);
    auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + MP.getvalue<int>(Piro::MeshParams::CONSTANTNUM)];
    
    cd.type = 2; // CSR
    cd.rowpointers.assign(N + 1, 0.0);
    laplacian_collect.push_back(CD_GPU);
    laplacian_collect.push_back(CD_GPU);
    float norm = pow((l[0] / float(n[0] - 2)), 2);
    // Iterate over all grid points
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::CSR::stencils::laplacian_cd(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 2){
        Piro::CSR::stencils::laplacian_dw(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 1){
        Piro::CSR::stencils::laplacian_uw(cd, n, norm);
    }
    
    cl_int err;
    laplacian_collect[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * cd.rowpointers.size(), 
        cd.rowpointers.data(), &err);

    laplacian_collect[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) *  cd.columns.size(), 
        cd.columns.data(), &err);

    laplacian_collect[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) *  cd.values.size(), 
        cd.values.data(), &err);
    
    Piro::INIT::getInstance().LAP_INIT = true;

    CDGPU.setvalue(Piro::CellDataGPU::LAPLACIAN_CSR, laplacian_collect);
    Piro::Equation::getInstance().sparsecount += cd.values.size();
    return 0;
}

int Piro::DENSE::laplacian(){
    Piro::CellData CD;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    std::vector<float> delta = SP.getvalue<std::vector<float>>(Piro::SolveParams::DELTA);
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    size_t total_size = n[0] * n[1] * n[2];
    float norm = pow((l[0] / float(n[0] - 2)), 2);
    auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) 
                        + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + MP.getvalue<int>(Piro::MeshParams::CONSTANTNUM)];
    
    // MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
    cd.type = 4; // DENSE
    cd.matrix.resize(total_size * total_size);
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::DENSE::stencils::laplacian_cd(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 2){
        Piro::DENSE::stencils::laplacian_dw(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 1){
        Piro::DENSE::stencils::laplacian_uw(cd, n, norm);
    }
    
    // put the matrix into CL buffer
    cl_int err;
    Piro::CLBuffer laplacian;
    Piro::kernels& kernels = Piro::kernels::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    laplacian.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * cd.matrix.size(), 
        cd.matrix.data(),
        &err);

    Piro::INIT::getInstance().LAP_INIT = true;
    CDGPU.setvalue(Piro::CellDataGPU::LAPLACIAN_DENSE, laplacian);
    return 0;
}

int Piro::COO::laplacian(){
    Piro::CLBuffer CD_GPU;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    Piro::CellData CD;
    std::vector<CLBuffer> laplacian_collect;
    // MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
    laplacian_collect.push_back(CD_GPU);
    auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) 
                        + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + MP.getvalue<int>(Piro::MeshParams::CONSTANTNUM)];
    
    cd.type = 3; // COO
    // cd.rowpointers.assign(N + 1, 0.0);
    laplacian_collect.push_back(CD_GPU);
    laplacian_collect.push_back(CD_GPU);
    float norm = pow((l[0] / float(n[0] - 2)), 2);
    // Iterate over all grid points
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::COO::stencils::laplacian_cd(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 2){
        Piro::COO::stencils::laplacian_dw(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 1){
        Piro::COO::stencils::laplacian_uw(cd, n, norm);
    }
    
    cl_int err;
    laplacian_collect[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * cd.rows.size(), 
        cd.rows.data(), &err);

    laplacian_collect[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) *  cd.columns.size(), 
        cd.columns.data(), &err);

    laplacian_collect[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) *  cd.values.size(), 
        cd.values.data(), &err);
    
    Piro::INIT::getInstance().LAP_INIT = true;

    CDGPU.setvalue(Piro::CellDataGPU::LAPLACIAN_COO, laplacian_collect);
    return 0;
}

int Piro::HT::laplacian(){
    Piro::CLBuffer CD_GPU;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    int N = n[0] * n[1] * n[2];
    Piro::CellData CD;
    auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) 
                        + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + MP.getvalue<int>(Piro::MeshParams::CONSTANTNUM)];
    // MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
    cd.type = 2; // CSR
    cd.rowpointers.assign(N + 1, 0.0);
    float norm = pow((l[0] / float(n[0] - 2)), 2);
    // Iterate over all grid points
    if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 3){
        Piro::HT::stencils::laplacian_cd(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 2){
        Piro::HT::stencils::laplacian_dw(cd, n, norm);
    }
    else if(SP.getvalue<int>(Piro::SolveParams::SPACESCHEME) == 1){
        Piro::HT::stencils::laplacian_uw(cd, n, norm);
    }
    Piro::Equation::getInstance().sparsecount += cd.values.size();
    Piro::logger::debug("Generating hash table");
    
    float load = SP.getvalue<float>(Piro::SolveParams::A) * pow(N, SP.getvalue<float>(Piro::SolveParams::B)) + SP.getvalue<float>(Piro::SolveParams::C);
    SP.setvalue(Piro::SolveParams::TABLE_SIZE, (int)(Piro::Equation::getInstance().sparsecount / (load * SP.getvalue<float>(Piro::SolveParams::LOADFACTOR))));
    
    Piro::logger::debug("Hash Table size : ", SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE));
    cd.Hash_keys_V.assign(SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE), -1);
    cd.Hash_val_V.assign(SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE), 0.0f);
    
    
    int ind;
    for (int i = 0; i < N; i++){
        int start = cd.rowpointers[i];
        int end = cd.rowpointers[i + 1];
        for(int j = start; j < end; j++){
            int col = cd.columns[j];
            int row = i;

            ind = row * N + col;
            Piro::methods::sethash(ind, cd.values[j], SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE), cd.Hash_keys_V, cd.Hash_val_V);
        }
    }
    cl_int err;
    cl_event event0, event1;
    std::vector<CLBuffer> laplacian_ht;
    laplacian_ht.push_back(CD_GPU);
    laplacian_ht.push_back(CD_GPU);

    laplacian_ht[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_WRITE , sizeof(int) * SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE), nullptr, &err);
    laplacian_ht[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_WRITE , sizeof(float) * SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE), nullptr, &err);
    
    
    CDGPU.setvalue(Piro::CellDataGPU::LAPLACIAN_HT, laplacian_ht);
    // hash_to_dense_and_print(Hash_keys_V, Hash_val_V, N, TABLE_SIZE);
    err = clEnqueueWriteBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_HT)[0].buffer, CL_FALSE, 
                                0, 
                                sizeof(int) * SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE),
                                cd.Hash_keys_V.data(), 
                                0, nullptr, &event0);
    // assert(Lap_rowptr_V.data() + rowouter != nullptr);
    
    err = clEnqueueWriteBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_HT)[1].buffer, CL_FALSE, 
                                0, 
                                sizeof(float) * SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE),
                                cd.Hash_val_V.data(), 
                                0, nullptr, &event1);

    // Wait for all transfers to complete
    clWaitForEvents(2, (cl_event[]){event0, event1});
    Piro::INIT::getInstance().LAP_INIT = true;
    return 0;
}