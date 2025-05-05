#include <iostream>
#include <vector>
#include <preprocess.hpp>
#include <init.hpp>
#include <inireader.hpp>
#include <extras.hpp>
#include <stringutilities.hpp>
#include <bc.hpp>
#include <solve.hpp>
#include <algorithm>
#include <CL/opencl.h>

std::vector<std::vector<int>> indices(6, std::vector<int>());
std::vector<int> indices_toprint;
std::vector<int> indices_toprint_vec;
std::vector<std::string> BC_property;
std::vector<float> BC_value;
Piro::Solve GS;
uint Q;
cl_mem memD, memE;

void opencl_initBC(){
    int N = MP.n[0] * MP.n[1] * MP.n[2];
    Q = Piro::vector_operations::flattenvector(indices).size();
    memE = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(uint) * Q, Piro::vector_operations::flattenvector(indices).data(), &err);
    memD = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N, MP.AMR[0].CD[0].values.data(), &err);
    if (err != CL_SUCCESS){
        std::cout << "BC error" << std::endl;
        }
}

void opencl_setBC(int ind){
    // int N = MP.n[0] * MP.n[1] * MP.n[2];
    size_t globalWorkSizeBC[1] = { (size_t)Q };
    // std::vector<float> prop = MP.AMR[0].CD[ind].values;

    // err = clEnqueueReadBuffer(queue, CDGPU.values_gpu[ind].buffer, CL_TRUE, 0,
    //          sizeof(float) * N, prop.data(), 0, NULL, NULL);
    // std::cout << "before setting BC" << std::endl;
    // printVector(prop);

    err |= clSetKernelArg(kernelBC, 0, sizeof(cl_mem), &CDGPU.values_gpu[ind].buffer);
    err |= clSetKernelArg(kernelBC, 1, sizeof(cl_mem), &memD);
    err |= clSetKernelArg(kernelBC, 2, sizeof(cl_mem), &memE);
    err |= clSetKernelArg(kernelBC, 3, sizeof(cl_uint), &Q);

    err = clEnqueueNDRangeKernel(queue, kernelBC, 1, NULL, globalWorkSizeBC, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        std::cout << "BC error" << std::endl;
        }
    
    // err = clEnqueueReadBuffer(queue, CDGPU.values_gpu[ind].buffer, CL_TRUE, 0,
    //           sizeof(float) * N, prop.data(), 0, NULL, NULL);
    // std::cout << "after setting BC" << std::endl;
    // printVector(prop);
    
}

void setbc(){
    for (int ind = 0; ind < 6; ind++){
        for(uint faces = 0; faces < indices[ind].size(); faces++){
            int msv = GS.matchscalartovar(BC_property[ind]);
            MP.AMR[0].CD[msv].values[indices[ind][faces]] = BC_value[ind];
        }
    }
}

void prepbc(){
    std::cout << "Preparing cells to print" << std::endl;
    for(uint ind = 0; ind < MP.n[0] * MP.n[1] * MP.n[2]; ind++){
        uint kd = ind / (MP.n[1] * MP.n[0]);
        uint jd = (ind / MP.n[0]) % MP.n[1];
        uint id = ind % MP.n[0];
        if (id == 0 || id == MP.n[0] - 1 || jd == 0 || jd == MP.n[1] - 1 || kd == 0 || kd == MP.n[2] - 1){
            indices_toprint.push_back(ind);
            indices_toprint_vec.push_back(ind);
            indices_toprint_vec.push_back(ind + MP.n[0]*MP.n[1]*MP.n[2]);
            indices_toprint_vec.push_back(ind + 2 * MP.n[0]*MP.n[1]*MP.n[2]);
            
        }
    }
    std::sort(indices_toprint.rbegin(), indices_toprint.rend());
    std::sort(indices_toprint_vec.rbegin(), indices_toprint_vec.rend());
}

void initbc(){
    std::cout << "Initialising boundary conditions" << std::endl;
    std::vector<int> BC_type;
    
    for(uint ind = 0; ind < MP.n[0] * MP.n[1] * MP.n[2]; ind++){
        uint kd = ind / (MP.n[1] * MP.n[0]);
        uint jd = (ind % (MP.n[1] * MP.n[0])) / MP.n[0];
        uint id = ind % MP.n[0];
        
        if(jd == 0 && (id != 0 && id != MP.n[0] - 1) && (kd != 0 && kd != MP.n[2] - 1)){
            // 3 : XZ plane
            indices[3].push_back(ind);
        }
        else if(jd == MP.n[1] - 1 && (id != 0 && id != MP.n[0] - 1) && (kd != 0 && kd != MP.n[2] - 1)){
            // 2 : XZ plane + breadth
            indices[2].push_back(ind);
        }
        
        if(kd == 0 && (id != 0 && id != MP.n[0] - 1) && (jd != 0 && jd != MP.n[1] - 1)){
            // 4 : XY plane 
            indices[4].push_back(ind);
        }
        else if(kd == MP.n[2] - 1 && (id != 0 && id != MP.n[0] - 1) && (jd != 0 && jd != MP.n[1] - 1)){
            // 5 : XY plane + height
            indices[5].push_back(ind);
        }

        if(id == 0 && (jd != 0 && jd != MP.n[1] - 1) && (kd != 0 && kd != MP.n[2] - 1)){
            // 1 : YZ plane
            indices[1].push_back(ind);
        }
        else if(id == MP.n[0] - 1 && (jd != 0 && jd != MP.n[1] - 1) && (kd != 0 && kd != MP.n[2] - 1)){
            // 0 : YZ plane + length
            indices[0].push_back(ind);
        }
    }

    IniReader reader(current_path.string() + "/assets/setup.ini");
    BC_type = Piro::string_utilities::convertStringVectorToInt(Piro::string_utilities::splitString(reader.get("BC", "type", "default_value"), ' '));
    BC_property = Piro::string_utilities::splitString(reader.get("BC", "property", "default_value"), ' ');
    BC_value = Piro::string_utilities::convertStringVectorToFloat(Piro::string_utilities::splitString(reader.get("BC", "values", "default_value"), ' '));
    setbc();
    prepbc();

    opencl_initBC();
    std::cout << "Boundary conditions initialised" << std::endl;
}

void readbc(){
    std::cout << "Reading boundary conditions" << std::endl;
    IniReader reader(current_path.string() + "/assets/setup.ini");

    if(Piro::string_utilities::countSpaces(reader.get("BC", "type", "default_value")) > 1){
        initbc();
    }

}
