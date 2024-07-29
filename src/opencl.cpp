#define CL_HPP_TARGET_OPENCL_VERSION 200 
#include <iostream>
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#elif _WIN32
    #include "../dependencies/include/CL/opencl.h"
#else
    #include "../dependencies/include/CL/opencl.h"
#endif
#include "../dependencies/include/gpuinit.hpp"
#include "../dependencies/include/kernel.cl.h"
#include "../dependencies/include/extras.hpp"
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/datatypes.hpp"
#include "../dependencies/include/bc.hpp"

float* B_ptr;
cl_int err;
// cl_platform_id platform;
cl_uint platformCount;
cl_uint num_devices;
cl_device_id *devices;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program  program_addVec,
            program_subtractVec,
            program_multiplyVec, 
            program_divideVec, 
            program_laplacian, 
            program_setBC;
cl_kernel   kernel_addVec,
            kernel_subtractVec,
            kernel_multiplyVec,
            kernel_divideVec,
            kernellaplacian,
            kernelBC;
cl_mem memBx, memCx, memDx, memEx;

static void print_device_info(cl_device_id device){
    char name[128];
    char vendor[128];

    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    std::cout << name << std::endl;
    std::cout << vendor << std::endl;

}

int opencl_init(){
    std::cout << "Initialising OpenCL" << std::endl;
     // Initialize OpenCL
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    
    // Get number of devices
    err = clGetDeviceIDs(platforms[DP.platformid], CL_DEVICE_TYPE_ALL, 3, NULL, &num_devices);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get number of devices" << std::endl;
        return 1;
    }

    // Allocate memory for devices array
    devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));

    // Get all device IDs
    err = clGetDeviceIDs(platforms[DP.platformid], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get device IDs" << std::endl;
        free(devices);
        return 1;
    }

    // Set default device
    std::cout << "Active device" << std::endl;
    device = devices[DP.id];
    print_device_info(device);

    // Initialize OpenCL
    std::cout << "Calling OpenCL" << std::endl;
    // Create OpenCL context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context" << std::endl;
        return 1;
    }

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue" << std::endl;
        clReleaseContext(context);
        return 1;
    }
    
    return 0;
}

cl_program opencl_CreateProgram(const char* dialog){
    cl_program program = clCreateProgramWithSource(context, 1, &dialog, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program with source" << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
    return program;
}

cl_int opencl_BuildProgram(cl_program program){
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program" << std::endl;
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        std::cerr << "Build log:\n" << log << std::endl;
        free(log);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    return err;
}

int opencl_build(){
     // Create program objects
    std::cout << "Building program : " << std::endl;
    program_addVec = opencl_CreateProgram(addVectors);
    program_multiplyVec = opencl_CreateProgram(multiplyVectors);
    program_subtractVec = opencl_CreateProgram(subtractVectors);
    program_divideVec = opencl_CreateProgram(divideVectors);
    program_laplacian = opencl_CreateProgram(laplaciancalc);
    program_setBC = opencl_CreateProgram(setBC);
    err = opencl_BuildProgram(program_addVec);
    err = opencl_BuildProgram(program_subtractVec);
    err = opencl_BuildProgram(program_multiplyVec);
    err = opencl_BuildProgram(program_divideVec);
    err = opencl_BuildProgram(program_laplacian);
    err = opencl_BuildProgram(program_setBC);
    
    std::cout << "Creating kernel" << std::endl;
    kernel_addVec = clCreateKernel(program_addVec, "addVectors", &err);
    kernel_multiplyVec = clCreateKernel(program_multiplyVec, "multiplyVectors", &err);
    kernel_subtractVec = clCreateKernel(program_subtractVec, "subtractVectors", &err);
    kernel_divideVec = clCreateKernel(program_divideVec, "divideVectors", &err);
    kernelBC = clCreateKernel(program_setBC, "setBC", &err);
    kernellaplacian = clCreateKernel(program_laplacian, "laplacian", &err);

    return 0;
}

int opencl_cleanup(){
    
    std::cout << "Cleanup started" << std::endl;
    // Clean up
    free(devices);
    // Cleanup
    clReleaseMemObject(memBx);
    clReleaseMemObject(memCx);
    clReleaseMemObject(memDx);
    clReleaseMemObject(memEx);
    clReleaseKernel(kernellaplacian);
    clReleaseKernel(kernel_addVec);
    clReleaseKernel(kernel_subtractVec);
    clReleaseKernel(kernel_multiplyVec);
    clReleaseKernel(kernel_divideVec);
    clReleaseKernel(kernelBC);
    clReleaseProgram(program_laplacian);
    clReleaseProgram(program_setBC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::cout << "Cleanup ended" << std::endl;
    return 0;
}
