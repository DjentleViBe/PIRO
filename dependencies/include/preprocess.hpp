#ifndef preprocess_hpp
#define preprocess_hpp

#include <datatypes.hpp>
#include <vector>
#include <string>
#include <CL/opencl.h>

extern Piro::MeshParams MP;
extern Piro::DeviceParams DP;
extern int ts;
extern bool LAP_INIT;
extern bool RHS_INIT;

namespace Piro{
    int laplacian_CSR_init();
    int preprocess(const std::string& name);
    void print_device_info(cl_device_id device);
    cl_int opencl_BuildProgram(cl_program program);
    cl_program opencl_CreateProgram(const char* dialog);
    int opencl_init();
    int opencl_build();
    int opencl_run();
    std::vector<unsigned char> readBinaryFile(const std::string& filePath, size_t& binarySize);
    int opencl_cleanup();
    void loadKernelsFromFile(const std::string& filename, 
        std::vector<std::string>& knm, 
        std::vector<std::string>& ksm);
    std::string readFile(const std::string& kernelName);
    std::vector<unsigned char> readBinaryFile(const std::string& filename);
}

#endif