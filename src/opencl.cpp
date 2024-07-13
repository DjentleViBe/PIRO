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

float* B_ptr;

static void print_device_info(cl_device_id device){
    char name[128];
    char vendor[128];

    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    std::cout << name << std::endl;
    std::cout << vendor << std::endl;

}

int opencl_call(float* hostA, float* hostB, int time, int N, int M, int P){
    
    cl_int err;
    cl_platform_id platform;
    cl_uint num_devices;
    cl_device_id *devices;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem memA, memB, memC;
    
    // Initialize OpenCL
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get platform IDs" << std::endl;
        return 1;
    }
    
    // Get number of devices
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get number of devices" << std::endl;
        return 1;
    }

    // Allocate memory for devices array
    devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));

    // Get all device IDs
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get device IDs" << std::endl;
        free(devices);
        return 1;
    }

    // Print information for each device
    // for (cl_uint i = 0; i < num_devices; ++i) {
    //   std::cout << "Device #" << i + 1 << std::endl;
    //    print_device_info(devices[i]);
    //}
    // Set default device
    std::cout << "Active device" << std::endl;
    device = devices[DP.id];
    
    // Initialize OpenCL
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get platform IDs" << std::endl;
        return 1;
    }

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
    
    // Build program
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
    std::cout << "Creating kernel" << std::endl;
    kernel = clCreateKernel(program, "matrixMultiply", &err);
    
    std::cout << "Creating buffers" << std::endl;
    // Create buffer for input data
    memA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N * M, hostA, &err);
    memB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * M * P, hostB, &err);
    memC = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * M * P, hostB, &err);
    // Set kernel arguments
    std::cout << "Buffers created" << std::endl;

    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &memA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memC);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &N);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &M);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &P);

    size_t globalWorkSize[2] = { (size_t)P, (size_t)N };  // Number of work-items
    std::cout << "matmulstarted" << std::endl;
    print_time();
    for(int ti = 0; ti < time; ti++){
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        
    }
    std::cout << "matmulend" << std::endl;
    print_time();
    err = clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                              sizeof(float) * N * P, hostB, 0, NULL, NULL);
    // Clean up
    free(devices);
    // Cleanup
    clReleaseMemObject(memA);
    clReleaseMemObject(memB);
    // clReleaseMemObject(memC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
