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

// math opertions programs and kernels
// 0: add
// 1: subtract
// 2: multiply
// 3: divide
std::vector<cl_program> program_math(9);
std::vector<cl_kernel> kernel_math(9);
std::vector<const char*> kernelSources = {addVectors, subtractVectors, multiplyVectors, divideVectors, 
                                        addVectors_constant, 
                                        subtractVectors_constant, 
                                        multiplyVectors_constant,
                                        divideVectors_constant,
                                        subtractVectors_self};
std::vector<const char*> kernelNames = {"addVectors", "subtractVectors", "multiplyVectors", "divideVectors",
                                        "addVectors_constant", 
                                        "subtractVectors_constant", 
                                        "multiplyVectors_constant",
                                        "divideVectors_constant",
                                        "subtractVectors_self"};

cl_program  program_gradient_type1,
            program_gradient_type2, 
            program_gradient_type3, 
            program_gradient_type4, 
            program_laplacian_scalar,
            program_sparseMatrixMultiply_CSR,
            program_laplaciansparseMatrixMultiply_CSR,
            program_laplacian_vector,
            program_setBC;
cl_kernel   kernelgradient_type1,
            kernelgradient_type2,
            kernelgradient_type3,
            kernelgradient_type4,
            kernellaplacianscalar,
            kernelsparseMatrixMultiplyCSR,
            kernellaplaciansparseMatrixMultiplyCSR,
            kernellaplacianvector,
            kernelBC;
cl_mem memBx, memCx, memDx, memEx;

static void print_device_info(cl_device_id device){
    char name[128];
    char vendor[128];

    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    std::cout << name << std::endl;
    std::cout << vendor << std::endl;

    cl_uint numComputeUnits;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numComputeUnits), &numComputeUnits, NULL);

    cl_uint maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);

    std::cout << "Compute Units: " << numComputeUnits << std::endl;
    std::cout << "Max Work Group Size: " << maxWorkGroupSize << std::endl;


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
    for(size_t i = 0; i < program_math.size(); i++){
        program_math[i] = opencl_CreateProgram(kernelSources[i]);
        err = opencl_BuildProgram(program_math[i]);
        kernel_math[i] = clCreateKernel(program_math[i], kernelNames[i], &err);
    }
    program_gradient_type1 = opencl_CreateProgram(gradcalc1);
    program_gradient_type2 = opencl_CreateProgram(gradcalc2);
    program_gradient_type3 = opencl_CreateProgram(gradcalc3);
    program_gradient_type4 = opencl_CreateProgram(gradcalc4);
    program_laplacian_scalar = opencl_CreateProgram(laplacianscalar);
    program_sparseMatrixMultiply_CSR = opencl_CreateProgram(sparseMatrixMultiplyCSR);
    program_laplaciansparseMatrixMultiply_CSR = opencl_CreateProgram(LaplaciansparseMatrixMultiplyCSR);
    program_laplacian_vector = opencl_CreateProgram(laplacianvector);
    program_setBC = opencl_CreateProgram(setBC);
    
    err = opencl_BuildProgram(program_gradient_type1);
    err = opencl_BuildProgram(program_gradient_type2);
    err = opencl_BuildProgram(program_gradient_type3);
    err = opencl_BuildProgram(program_gradient_type4);
    err = opencl_BuildProgram(program_laplacian_scalar);
    err = opencl_BuildProgram(program_sparseMatrixMultiply_CSR);
    err = opencl_BuildProgram(program_laplaciansparseMatrixMultiply_CSR);
    err = opencl_BuildProgram(program_laplacian_vector);
    err = opencl_BuildProgram(program_setBC);

    if (err != CL_SUCCESS){
        std::cout << "program error" << std::endl;
    }
    
    std::cout << "Creating kernel" << std::endl;
    kernelBC = clCreateKernel(program_setBC, "setBC", &err);
    kernellaplacianscalar = clCreateKernel(program_laplacian_scalar, "laplacianscalar", &err);
    kernelsparseMatrixMultiplyCSR = clCreateKernel(program_sparseMatrixMultiply_CSR, "sparseMatrixMultiplyCSR", &err);
    kernellaplaciansparseMatrixMultiplyCSR = clCreateKernel(program_laplaciansparseMatrixMultiply_CSR, "LaplaciansparseMatrixMultiplyCSR", &err);
    kernellaplacianvector = clCreateKernel(program_laplacian_vector, "laplacianvector", &err);
    kernelgradient_type1 = clCreateKernel(program_gradient_type1, "gradient1", &err);
    kernelgradient_type2 = clCreateKernel(program_gradient_type2, "gradient2", &err);
    kernelgradient_type3 = clCreateKernel(program_gradient_type3, "gradient3", &err);
    kernelgradient_type4 = clCreateKernel(program_gradient_type4, "gradient4", &err);

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
    clReleaseKernel(kernelgradient_type1);
    clReleaseKernel(kernelgradient_type2);
    clReleaseKernel(kernelgradient_type3);
    clReleaseKernel(kernelgradient_type4);
    clReleaseKernel(kernellaplacianscalar);
    clReleaseKernel(kernellaplacianvector);
    clReleaseKernel(kernelsparseMatrixMultiplyCSR);
    clReleaseKernel(kernellaplaciansparseMatrixMultiplyCSR);
    for(size_t i = 0; i < program_math.size(); i++){
        clReleaseKernel(kernel_math[i]);
        clReleaseProgram(program_math[i]);
    }
    clReleaseKernel(kernelBC);
    clReleaseProgram(program_gradient_type1);
    clReleaseProgram(program_gradient_type2);
    clReleaseProgram(program_gradient_type3);
    clReleaseProgram(program_gradient_type4);
    clReleaseProgram(program_laplacian_scalar);
    clReleaseProgram(program_sparseMatrixMultiply_CSR);
    clReleaseProgram(program_laplaciansparseMatrixMultiply_CSR);
    clReleaseProgram(program_laplacian_vector);
    clReleaseProgram(program_setBC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::cout << "Cleanup ended" << std::endl;
    return 0;
}
