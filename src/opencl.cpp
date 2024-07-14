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

static void print_device_info(cl_device_id device){
    char name[128];
    char vendor[128];

    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    std::cout << name << std::endl;
    std::cout << vendor << std::endl;

}

int opencl_call(float* hostA, float* hostB, int time, uint N, uint M, uint P){
    
    cl_int err;
    cl_platform_id platform;
    cl_uint num_devices;
    cl_device_id *devices;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program, program_setBC;
    cl_kernel kernel, kernelBC;
    cl_mem memA, memB, memC, memD, memE;
    
    // Initialize OpenCL
    cl_uint platformCount;
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    /*
    for (cl_platform_id platform : platforms) {
        size_t size;
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &size);
        std::string name(size, '\0');
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, &name[0], nullptr);
        std::cout << "Platform: " << name << std::endl;
    }*/
    
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

    // Print information for each device
    for (cl_uint i = 0; i < num_devices; ++i) {
       std::cout << "Device #" << i + 1 << std::endl;
        print_device_info(devices[i]);
    }
    // Set default device
    std::cout << "Active device" << std::endl;
    device = devices[DP.id];
    
    // Initialize OpenCL

    print_device_info(device);
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

    // Create program object
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program with source" << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    program_setBC = clCreateProgramWithSource(context, 1, &setBC, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program with source" << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    
    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    err = clBuildProgram(program_setBC, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program" << std::endl;
        size_t log_size;
        clGetProgramBuildInfo(program_setBC, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program_setBC, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        std::cerr << "Build log:\n" << log << std::endl;
        free(log);
        clReleaseProgram(program_setBC);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    std::cout << "Creating kernel" << std::endl;
    kernel = clCreateKernel(program, "matrixMultiply", &err);
    kernelBC = clCreateKernel(program_setBC, "setBC", &err);

    std::cout << "Creating buffers" << std::endl;
    // Create buffer for input data
    memA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N * M, hostA, &err);
    memB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * M * P, hostB, &err);
    memC = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * M * P, hostB, &err);
    memD = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N * P, MP.AMR[0].CD[0].values.data(), &err);
    
    uint Q = flattenvector(indices).size();
    memE = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(uint) * Q, flattenvector(indices).data(), &err);
    // Set kernel arguments
    std::cout << "Buffers created" << std::endl;
    // std::vector<float> hostC(N * P);
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &memA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memC);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &N);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &M);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &P);

    
    err |= clSetKernelArg(kernelBC, 0, sizeof(cl_mem), &memB);
    err |= clSetKernelArg(kernelBC, 1, sizeof(cl_mem), &memD);
    err |= clSetKernelArg(kernelBC, 2, sizeof(cl_mem), &memE);
    err |= clSetKernelArg(kernelBC, 3, sizeof(cl_uint), &Q);

    size_t globalWorkSize[2] = { (size_t)P, (size_t)N };  // Number of work-items
    size_t globalWorkSizeBC[1] = { (size_t)Q };  // Number of work-items
    std::cout << "matmulstarted" << std::endl;
    print_time();
    int totaliter = SP.totaltime / SP.timestep;
    for(int ti = 0; ti < totaliter; ti++){
        std::cout << ti << std::endl;
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        err = clEnqueueCopyBuffer(queue, memC, memB, 0, 0, sizeof(float) * N * P, 0, NULL, NULL);
        // set boundary condition
        err = clEnqueueNDRangeKernel(queue, kernelBC, 1, NULL, globalWorkSizeBC, NULL, 0, NULL, NULL);
        
    }
    std::cout << "matmulend" << std::endl;
    print_time();
    err = clEnqueueReadBuffer(queue, memB, CL_TRUE, 0,
                              sizeof(float) * N * P, hostB, 0, NULL, NULL);
    
    
    // Clean up
    free(devices);
    // Cleanup
    clReleaseMemObject(memA);
    clReleaseMemObject(memB);
    clReleaseMemObject(memC);
    clReleaseMemObject(memD);
    // clReleaseMemObject(memC);
    clReleaseKernel(kernel);
    clReleaseKernel(kernelBC);
    clReleaseProgram(program);
    clReleaseProgram(program_setBC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
