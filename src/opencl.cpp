#define CL_HPP_TARGET_OPENCL_VERSION 200 
#include <iostream>
#include <CL/opencl.h>
#include <vector>
#include <gpuinit.hpp>
#include <kernel.cl.h>
#include <kernels.h>
#include <extras.hpp>
#include <preprocess.hpp>
#include <openclutilities.hpp>
#include <datatypes.hpp>
#include <bc.hpp>
#include <vector>
#include <logger.hpp>

float* B_ptr;
cl_int err;
// cl_platform_id platform;
cl_uint platformCount;
cl_uint num_devices;
cl_device_id *devices;
cl_device_id device;
cl_context Piro::kernels::context;
cl_command_queue Piro::kernels::queue;
cl_uint Piro::kernels::maxWorkGroupSize;

namespace Piro::kernels {
    std::vector<cl_program> program_math(9);
    std::vector<cl_kernel> kernel_math(9);
    std::vector<cl_program> program(14);
    std::vector<cl_kernel> kernel(14);
}

std::vector<const char*> kernelSourcesmath = {addVectors, subtractVectors, multiplyVectors, divideVectors, 
                                                addVectors_constant, 
                                                subtractVectors_constant, 
                                                multiplyVectors_constant,
                                                divideVectors_constant,
                                                subtractVectors_self};
std::vector<const char*> kernelNamesmath = {"addVectors", "subtractVectors", "multiplyVectors", "divideVectors",
                                            "addVectors_constant", 
                                            "subtractVectors_constant", 
                                            "multiplyVectors_constant",
                                            "divideVectors_constant",
                                            "subtractVectors_self",
                                            "setBC"};
std::vector<const char*> kernelSources = {setBC,
                                        gradcalc1,
                                        gradcalc2, 
                                        gradcalc3, 
                                        gradcalc4, 
                                        laplacianscalar,
                                        lu_decompose_dense,
                                        sparseMatrixMultiplyCSR,
                                        laplaciansparseMatrixMultiplyCSR,
                                        laplacianvector,
                                        filter_array,
                                        filter_row,
                                        forward_substitution_csr,
                                        backward_substitution_csr};
std::vector<const char*> kernelNames = {"setBC",
                                        "gradcalc1",
                                        "gradcalc2", 
                                        "gradcalc3", 
                                        "gradcalc4", 
                                        "laplacianscalar",
                                        "lu_decompose_dense",
                                        "sparseMatrixMultiplyCSR",
                                        "laplaciansparseMatrixMultiplyCSR",
                                        "laplacianvector",
                                        "filter_array",
                                        "filter_row",
                                        "forward_substitution_csr",
                                        "backward_substitution_csr"};
cl_mem memBx, memCx, memDx, memEx;

void Piro::print_device_info(cl_device_id device){
    char name[128];
    char vendor[128];

    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    Piro::logger::info(name);
    Piro::logger::info(vendor);

    cl_uint numComputeUnits;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numComputeUnits), &numComputeUnits, NULL);

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(Piro::kernels::maxWorkGroupSize), &Piro::kernels::maxWorkGroupSize, NULL);

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &Piro::kernels::maxWorkGroupSize, NULL);
    Piro::logger::info("Compute Units: ", numComputeUnits);
    Piro::logger::info("Max Work Group Size: ", Piro::kernels::maxWorkGroupSize);
    // std::cout << "Max Global Mem Size: " << globalMemSize << " bytes" << std::endl;
    // std::cout << "Max Alloc Size: " << maxAllocSize << " bytes" << std::endl;
}

int Piro::opencl_init(){
    Piro::logger::info("Initialising OpenCL");
    // Initialize OpenCL
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    
    // Get number of devices
    err = clGetDeviceIDs(platforms[DP.platformid], CL_DEVICE_TYPE_ALL, 3, NULL, &num_devices);
    if (err != CL_SUCCESS) {
        Piro::logger::info("Failed to get number of devices");
        return 1;
    }

    // Allocate memory for devices array
    devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));

    // Get all device IDs
    err = clGetDeviceIDs(platforms[DP.platformid], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (err != CL_SUCCESS) {
        Piro::logger::info("Failed to get device IDs");
        free(devices);
        return 1;
    }

    // Set default device
    Piro::logger::info("Active device");
    device = devices[DP.id];
    Piro::print_device_info(device);

    // Initialize OpenCL
    Piro::logger::info("Calling OpenCL");
    // Create OpenCL context
    Piro::kernels::context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        Piro::logger::info("Failed to create OpenCL context");
        return 1;
    }

    // Create command queue
    Piro::kernels::queue = clCreateCommandQueue(Piro::kernels::context, device, 0, &err);
    if (err != CL_SUCCESS) {
        Piro::logger::info("Failed to create command queue");
        clReleaseContext(Piro::kernels::context);
        return 1;
    }
    
    return 0;
}

cl_program Piro::opencl_CreateProgram(const char* dialog){
    cl_program program = clCreateProgramWithSource(Piro::kernels::context, 1, &dialog, NULL, &err);
    if (err != CL_SUCCESS) {
        Piro::logger::info("Failed to create program with source");
        clReleaseCommandQueue(Piro::kernels::queue);
        clReleaseContext(Piro::kernels::context);
    }
    return program;
}

cl_int Piro::opencl_BuildProgram(cl_program program){
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        Piro::logger::info("Failed to build program");
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        Piro::logger::info("Build log:\n", log);
        free(log);
        clReleaseProgram(program);
        clReleaseCommandQueue(Piro::kernels::queue);
        clReleaseContext(Piro::kernels::context);
        return 1;
    }
    return err;
}

int Piro::opencl_build(){
    // Create program objects
    Piro::logger::info("Building program : initiated");
    for(size_t i = 0; i < Piro::kernels::program_math.size(); i++){
        Piro::kernels::program_math[i] = opencl_CreateProgram(kernelSourcesmath[i]);
        err = Piro::opencl_BuildProgram(Piro::kernels::program_math[i]);
        Piro::kernels::kernel_math[i] = clCreateKernel(Piro::kernels::program_math[i], kernelNamesmath[i], &err);
    }
    for(size_t i = 0; i < Piro::kernels::program.size(); i++){
        Piro::kernels::program[i] = opencl_CreateProgram(kernelSources[i]);
        err = Piro::opencl_BuildProgram(Piro::kernels::program[i]);
        Piro::kernels::kernel[i] = clCreateKernel(Piro::kernels::program[i], kernelNames[i], &err);
    }

    if (err != CL_SUCCESS){
        Piro::logger::info("program error");
    }
    Piro::logger::info("Building program : completed");
    return 0;
}

int Piro::opencl_cleanup(){
    
    Piro::logger::info("Cleanup started");
    // Clean up
    free(devices);
    // Cleanup
    clReleaseMemObject(memBx);
    clReleaseMemObject(memCx);
    clReleaseMemObject(memDx);
    clReleaseMemObject(memEx);

    for(size_t i = 0; i < Piro::kernels::program_math.size(); i++){
        clReleaseKernel(Piro::kernels::kernel_math[i]);
        clReleaseProgram(Piro::kernels::program_math[i]);
    }
    for(size_t i = 0; i < Piro::kernels::program.size(); i++){
        clReleaseKernel(Piro::kernels::kernel[i]);
        clReleaseProgram(Piro::kernels::program[i]);
    }
    clReleaseCommandQueue(Piro::kernels::queue);
    clReleaseContext(Piro::kernels::context);

    Piro::logger::info("Cleanup ended");
    return 0;
}

namespace Piro::opencl_utilities{
    template <typename T>
    std::vector<T> copyCL(cl_command_queue queue, cl_mem memC, int N, cl_event *event6) {
        std::vector<T> hostValues(N);
        clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                            sizeof(T) * N, hostValues.data(), 0, NULL, event6);
        clFinish(queue);
        return hostValues;
    }

    template <typename U>
    int copyCL_offset(cl_command_queue queue, cl_mem memC, std::vector<U>& Lap, int offset, int N, cl_event *event6) {
        size_t offset_size = sizeof(U) * offset;     
        clEnqueueReadBuffer(queue, memC, CL_FALSE, offset_size,
                            sizeof(U) * (N - offset), Lap.data() + offset, 0, NULL, event6);
        
        clWaitForEvents(1, event6);
        return 0;
    }

    template std::vector<float> copyCL<float>(cl_command_queue, cl_mem, int, cl_event*);
    template std::vector<int> copyCL<int>(cl_command_queue, cl_mem, int, cl_event*);

    template int copyCL_offset<float>(cl_command_queue, cl_mem, std::vector<float>& ,int, int, cl_event*);
    template int copyCL_offset<int>(cl_command_queue, cl_mem, std::vector<int>& ,int, int, cl_event*);

}