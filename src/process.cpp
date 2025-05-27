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
#include <CL/opencl.h>
#include <logger.hpp>
#include <openclutilities.hpp>
#include <matrixgenerations.hpp>
#include <kernelmethods.hpp>

using namespace Piro;

int process::matchscalartovar(std::string var){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    auto& cd = MP.getvalue<std::vector<AMR_LEVELS>>(Piro::MeshParams::AMRLEVELS)[0].amr[0];
    for(int v = 0; v < cd.CD.size(); v++){
        if(var == cd.CD[v].Scalars){
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

int process::matchvectortovar(std::string var){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    auto& cd = MP.getvalue<std::vector<AMR_LEVELS>>(Piro::MeshParams::AMRLEVELS)[0].amr[0];
    for(int v = 0; v < cd.CD.size(); v++){
        if(var == cd.CD[v].Scalars){
            return v;
        }
    }
    return 0;
}


CLBuffer process::ddt_r(std::string var){
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    int token = process::matchscalartovar(var);
    auto& vec = CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU);
    vec[token].ind = token;
    return vec[token];
}

std::vector<CLBuffer> process::laplacian(std::string var1, std::string var2){
    // int ind = matchscalartovar(var2);
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    int token2 = process::matchscalartovar(var2);
    if(INIT::getInstance().LAP_INIT == false){
        Piro::logger::info("Generating laplacian");
        // needs to be done just once until CDGPU.indices and CDGPU.values are filled in
        if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 0){
            Piro::matrix_generations::CSR::laplacian();
            auto& partA = CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[token2];
            auto& partB = CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_CSR);
            Piro::kernelmethods::csrscale({partA}, partB, 0);
            
            return {CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[0], 
                    CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[1], 
                    CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[2]};
        }
        else if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 1){
            Piro::matrix_generations::DENSE::laplacian();
            return {CDGPU.getvalue<Piro::CLBuffer>(Piro::CellDataGPU::LAPLACIAN_DENSE)};
        }
        else if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 2){
            Piro::matrix_generations::HT::laplacian();
            return {CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_HT)[0], 
                    CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_HT)[1]};
        }
        else if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 3){
            Piro::matrix_generations::COO::laplacian();
            return {CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_COO)[0], 
                    CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_COO)[1], 
                    CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_COO)[2]};
        }
        else{
            Piro::logger::info("Invalid DataType");
            exit(1);
        }

    }
    else{
        if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 0){
            return {CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[0], 
                    CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[1], 
                    CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[2]};
        }
        else if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 1){
            return {CDGPU.getvalue<Piro::CLBuffer>(Piro::CellDataGPU::LAPLACIAN_DENSE)};
        }
        else if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 2){
            return {CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_HT)[0], 
                    CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_HT)[1]};
        }
        else if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 3){
            return {CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_COO)[0], 
                    CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_COO)[1], 
                    CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_COO)[2]};
        }
        else{
            Piro::logger::info("Invlid DataType");
            exit(1);
        }
    }
}

std::vector<CLBuffer> process::gradient(std::string var1){
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    int token = process::matchscalartovar(var1);

    if(INIT::getInstance().GRAD_INIT == false){
        // Piro::logger::info("Generating gradient");
        if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 0){
            Piro::matrix_generations::CSR::gradient();
            auto& mat = CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::GRADIENT_CSR);
            mat[0].ind = token;
            return {mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]};
        }
        else{
            Piro::logger::info("Invalid DataType");
            exit(1);
        }
    }
    else{
        if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 0){
            auto& mat = CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::GRADIENT_CSR);
            return {mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]};
        }
        else{
            Piro::logger::info("Invalid DataType");
            exit(1);
        }
    }
}

std::vector<CLBuffer> process::vector(std::string var1){
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    int ind = process::matchscalartovar(var1);
    // Piro::logger::info("Generating gradient");
    if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 0){
        return {CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[ind]};
    }
    else{
        Piro::logger::info("Invalid DataType");
        exit(1);
    }
}

std::vector<CLBuffer> process::div(std::string var1, std::string var2){
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    // Piro::kernels& kernels = Piro::kernels::getInstance();
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    // var 1 requires linearisation if it is a function of var2, otherwise make a diagonal matrix of N with var1 and
    // dot it with DIV_CSR.
    // int N = n[0] * n[1] * n[2];
    // size_t globalWorkSize[1] = { (size_t)N };
    // int token1 = process::matchscalartovar(var1);
    int token2 = process::matchscalartovar(var2);
        
    // needs to be done just once until CDGPU.indices and CDGPU.values are filled in
    if(SP.getvalue<int>(Piro::SolveParams::DATATYPE) == 0){
        if(INIT::getInstance().DIV_INIT == false){
            Piro::logger::info("Generating divergence");
            Piro::matrix_generations::CSR::div();
            Piro::logger::info("Generating divergence : complete");
            auto& partA = CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[token2];
            auto& partB = CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::DIV_CSR);
            // auto& partC = CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[token1];
            // Piro::print_utilities::printCL(partA.buffer, N , 1);
            // element-wise multiplication of DIV_CSR with the diagonal matrix of token2
            Piro::kernelmethods::csrgeam_2({partA}, partB, 2);
            /*
            auto& cd = MP.getvalue<std::vector<AMR_LEVELS>>(Piro::MeshParams::AMRLEVELS)[0].amr[0];.CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + MP.getvalue<int>(Piro::MeshParams::CONSTANTNUM) + 2];
            int nnz = cd.values.size();
            std::vector<int> rp = Piro::opencl_utilities::copyCL<int>(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE),
                                            CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[0].buffer, N , NULL);
            std::vector<int> col = Piro::opencl_utilities::copyCL<int>(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE),
                                            CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[1].buffer, 3*nnz , NULL); 
            std::vector<float> val = Piro::opencl_utilities::copyCL<float>(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE),
                                            CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[2].buffer, 3*nnz , NULL);         
            Piro::print_utilities::csr_to_dense_and_print(rp, col, val, N);
            */

        }
        
        return {CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[0], 
                CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[1],
                CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::RHS)[2]};
    }
    else{
        Piro::logger::info("Invalid DataType");
        exit(1);
    }
}

scalarMatrix::scalarMatrix(CLBuffer SM) : smatrix(SM) {
    // No assignment operator is used here
}

void scalarMatrix::Solve(float currenttime){
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    float timestep = SP.getvalue<float>(Piro::SolveParams::TIMESTEP);
    int totaltimesteps = SP.getvalue<int>(Piro::SolveParams::TOTALTIMESTEPS);
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    auto& cd = MP.getvalue<std::vector<AMR_LEVELS>>(Piro::MeshParams::AMRLEVELS)[0].amr[0];
    cl_int err;
    int N = n[0] * n[1] * n[2];
    INIT::getInstance().ts = int(currenttime / timestep);
    Piro::logger::info("Timestep : ", INIT::getInstance().ts + 1, " / ", totaltimesteps);
    // apply Boundary Conditions
    err = clEnqueueCopyBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), smatrix.buffer, CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[1].buffer, 0, 0, sizeof(float) * N, 0, NULL, NULL);
    Piro::bc::opencl_setBC(1);
    // Piro::print_utilities::printCL(smatrix.buffer, N , 1);
    if((INIT::getInstance().ts + 1) % SP.getvalue<int>(Piro::SolveParams::SAVE) == 0){
        Piro::logger::info("Post processing started");
        // total number of vectors and scalars
        
        err = clEnqueueReadBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[1].buffer, CL_TRUE, 0,
                sizeof(float) * N, cd.CD[1].values.data(), 0, NULL, NULL);
        if (err != CL_SUCCESS){
            std::cout << "Solve error" << std::endl;
        }
        Piro::post::export_paraview(INIT::getInstance().ts);
        Piro::logger::info("Post processing finished\n");
    }
}