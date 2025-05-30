#include <iostream>
#include <vector>
#include <preprocess.hpp>
#include <init.hpp>
#include <fileutilities.hpp>
#include <extras.hpp>
#include <stringutilities.hpp>
#include <bc.hpp>
#include <process.hpp>
#include <algorithm>
#include <CL/opencl.h>
#include <logger.hpp>
#include <openclutilities.hpp>
#include <printutilities.hpp>

uint Q;
cl_mem memD, memE;

using namespace Piro;
void bc::opencl_initBC(){
    cl_int err;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::bc::indices& indval = Piro::bc::indices::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    int N = n[0] * n[1] * n[2];
    Piro::logger::info("OpenCL boundary initialisation");
    auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0];
    Q = Piro::vector_operations::flattenvector(indval.getvalue<std::vector<std::vector<int>>>(Piro::bc::indices::IND)).size();
    memE = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(uint) * Q, Piro::vector_operations::flattenvector(indval.getvalue<std::vector<std::vector<int>>>(Piro::bc::indices::IND)).data(), &err);
    memD = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N, cd.CD[1].values.data(), &err);
    if (err != CL_SUCCESS){
        std::cout << "BC error" << std::endl;
    }
}

void bc::opencl_setBC(int ind){
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    // int N = MP.n[0] * MP.n[1] * MP.n[2];
    size_t globalWorkSizeBC[1] = { (size_t)Q };
    // std::vector<float> prop = MP.AMR[0].CD[ind].values;

    // err = clEnqueueReadBuffer(queue, CDGPU.values_gpu[ind].buffer, CL_TRUE, 0,
    //          sizeof(float) * N, prop.data(), 0, NULL, NULL);
    // std::cout << "before setting BC" << std::endl;
    // printVector(prop);
    cl_int err = clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[0], 0, sizeof(cl_mem), 
                                &CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::VALUES_GPU)[ind].buffer);
    err = clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[0], 1, sizeof(cl_mem), &memD);
    err = clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[0], 2, sizeof(cl_mem), &memE);
    err = clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[0], 3, sizeof(cl_uint), &Q);

    err = clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[0], 1, NULL, globalWorkSizeBC, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        std::cout << "BC error" << std::endl;
        }
    
    // err = clEnqueueReadBuffer(queue, CDGPU.values_gpu[ind].buffer, CL_TRUE, 0,
    //           sizeof(float) * N, prop.data(), 0, NULL, NULL);
    // std::cout << "after setting BC" << std::endl;
    // printVector(prop);
    
}

void bc::setbc(){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::bc::indices& indval = Piro::bc::indices::getInstance();
    Piro::process GS;
    
    auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0];
    for (int ind = 0; ind < 6; ind++){
        for(uint faces = 0; faces < indval.getvalue<std::vector<std::vector<int>>>(Piro::bc::indices::IND)[ind].size(); faces++){
            int msv = GS.matchscalartovar(indval.getvalue<std::vector<std::string>>(Piro::bc::indices::BC_PROPERTY)[ind]);
            cd.CD[msv].values[indval.getvalue<std::vector<std::vector<int>>>(Piro::bc::indices::IND)[ind][faces]] = indval.getvalue<std::vector<float>>(Piro::bc::indices::BC_VALUE)[ind];
        }
    }
}

void bc::prepbc(){
    Piro::logger::info("Preparing cells to print");
    std::vector<int> indices_toprint;
    std::vector<int> indices_toprint_vec;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::bc::indices& indval = Piro::bc::indices::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    // int N = n[0] * n[1] * n[2];
    for(uint ind = 0; ind < n[0] * n[1] * n[2]; ind++){
        uint kd = ind / (n[1] * n[0]);
        uint jd = (ind / n[0]) % n[1];
        uint id = ind % n[0];
        if (id == 0 || id == n[0] - 1 || jd == 0 || jd == n[1] - 1 || kd == 0 || kd == n[2] - 1){
            indices_toprint.push_back(ind);
            indices_toprint_vec.push_back(ind);
            indices_toprint_vec.push_back(ind + n[0]*n[1]*n[2]);
            indices_toprint_vec.push_back(ind + 2 * n[0]*n[1]*n[2]);
            
        }
    }
    std::sort(indices_toprint.rbegin(), indices_toprint.rend());
    std::sort(indices_toprint_vec.rbegin(), indices_toprint_vec.rend());
    indval.setvalue(Piro::bc::indices::INDTOPRINT, indices_toprint);
    indval.setvalue(Piro::bc::indices::INDTOPRINTVEC, indices_toprint_vec);

}

void bc::initbc(){
    Piro::logger::info("Initialising boundary conditions");
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::bc::indices& indval = Piro::bc::indices::getInstance();
    
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<int> BC_type;
    std::vector<std::vector<int>> indices(6, std::vector<int>());
    for(uint ind = 0; ind < n[0] * n[1] * n[2]; ind++){
        uint kd = ind / (n[1] * n[0]);
        uint jd = (ind % (n[1] * n[0])) / n[0];
        uint id = ind % n[0];
        
        if(jd == 0 && (id != 0 && id != n[0] - 1) && (kd != 0 && kd != n[2] - 1)){
            // 3 : XZ plane
            indices[3].push_back(ind);
        }
        else if(jd == n[1] - 1 && (id != 0 && id != n[0] - 1) && (kd != 0 && kd != n[2] - 1)){
            // 2 : XZ plane + breadth
            indices[2].push_back(ind);
        }
        
        if(kd == 0 && (id != 0 && id != n[0] - 1) && (jd != 0 && jd != n[1] - 1)){
            // 4 : XY plane 
            indices[4].push_back(ind);
        }
        else if(kd == n[2] - 1 && (id != 0 && id != n[0] - 1) && (jd != 0 && jd != n[1] - 1)){
            // 5 : XY plane + height
            indices[5].push_back(ind);
        }

        if(id == 0 && (jd != 0 && jd != n[1] - 1) && (kd != 0 && kd != n[2] - 1)){
            // 1 : YZ plane
            indices[1].push_back(ind);
        }
        else if(id == n[0] - 1 && (jd != 0 && jd != n[1] - 1) && (kd != 0 && kd != n[2] - 1)){
            // 0 : YZ plane + length
            indices[0].push_back(ind);
        }
    }
    indval.setvalue(Piro::bc::indices::IND, indices);
    file_utilities::IniReader reader(Piro::file_utilities::current_path.string() + "/assets/setup.ini");
    indval.setvalue(Piro::bc::indices::BC_TYPE, Piro::string_utilities::convertStringVectorToInt(Piro::string_utilities::splitString(reader.get("BC", "type", "default_value"), ' ')));
    indval.setvalue(Piro::bc::indices::BC_PROPERTY, Piro::string_utilities::splitString(reader.get("BC", "property", "default_value"), ' '));
    indval.setvalue(Piro::bc::indices::BC_VALUE, Piro::string_utilities::convertStringVectorToFloat(Piro::string_utilities::splitString(reader.get("BC", "values", "default_value"), ' ')));
    
    Piro::bc::setbc();
    Piro::bc::prepbc();

    bc::opencl_initBC(); 
    Piro::logger::info("Boundary conditions initialised");
}

void bc::readbc(){
    Piro::logger::info("Reading boundary conditions");
    file_utilities::IniReader reader(Piro::file_utilities::current_path.string() + "/assets/setup.ini");

    if(Piro::string_utilities::countSpaces(reader.get("BC", "type", "default_value")) > 1){
        bc::initbc();
    }

}

bool bc::isBoundaryPoint(int x, int y, int z) {
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
    // Check if the point is on the boundary of any axis
    if (x == 0 || x == n[0] - 1 || y == 0 || y == n[1] - 1 || z == 0 || z == n[2] - 1) {
        return true;
    }
    return false;
}
