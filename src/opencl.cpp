#define CL_HPP_TARGET_OPENCL_VERSION 200 
#include <iostream>
#include <CL/opencl.h>
#include <vector>
#include <extras.hpp>
#include <preprocess.hpp>
#include <openclutilities.hpp>
#include <datatypes.hpp>
#include <bc.hpp>
#include <vector>
#include <logger.hpp>
#include <fstream>
#include <sstream>
#include <init.hpp>

cl_int err;
// cl_platform_id platform;
cl_uint platformCount;
cl_uint num_devices;
cl_device_id *devices;
cl_device_id device;

/*
namespace Piro::kernels {
    std::vector<cl_program> program_math(9);
    std::vector<cl_kernel> kernel_math(9);
    std::vector<cl_program> program(14);
    std::vector<cl_kernel> kernel(14);
}*/

std::vector<std::string> kernelSourcesmath;
std::vector<std::string> kernelNamesmath;
std::vector<std::string> kernelSources;
std::vector<std::string> kernelNames;
cl_mem memBx, memCx, memDx, memEx;

void Piro::print_device_info(cl_device_id device){
    Piro::kernels& kernels = Piro::kernels::getInstance();
    char name[128];
    char vendor[128];

    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    Piro::logger::info(name);
    Piro::logger::info(vendor);
    

    cl_uint numComputeUnits;
    cl_uint maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numComputeUnits), &numComputeUnits, NULL);

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(kernels.getvalue<cl_uint>(Piro::kernels::MAXWORKGROUPSIZE)), 
                    &maxWorkGroupSize, NULL);
    kernels.setvalue(Piro::kernels::MAXWORKGROUPSIZE, maxWorkGroupSize);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &kernels.getvalue<cl_uint>(Piro::kernels::MAXWORKGROUPSIZE), NULL);
    Piro::logger::info("Compute Units: ", numComputeUnits);
    Piro::logger::info("Max Work Group Size: ", kernels.getvalue<cl_uint>(Piro::kernels::MAXWORKGROUPSIZE));
    // std::cout << "Max Global Mem Size: " << globalMemSize << " bytes" << std::endl;
    // std::cout << "Max Alloc Size: " << maxAllocSize << " bytes" << std::endl;
}


std::string Piro::readFile(const std::string& kernelName) {
    std::ifstream file(current_path.string() + "/assets/kernels/" + kernelName + ".cl");  // Assume source file has the same name as the kernel
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel source file: " + current_path.string() + "/assets/kernels/" + kernelName + ".cl");
    }

    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

int Piro::opencl_init(){
    Piro::logger::info("Initialising OpenCL");
    Piro::DeviceParams& DP = Piro::DeviceParams::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    // Initialize OpenCL
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    
    // Get number of devices
    err = clGetDeviceIDs(platforms[DP.getvalue<int>(DeviceParams::PLATFORMID)], CL_DEVICE_TYPE_ALL, 3, NULL, &num_devices);
    if (err != CL_SUCCESS) {
        Piro::logger::info("Failed to get number of devices");
        return 1;
    }

    // Allocate memory for devices array
    devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));

    // Get all device IDs
    err = clGetDeviceIDs(platforms[DP.getvalue<int>(DeviceParams::PLATFORMID)], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (err != CL_SUCCESS) {
        Piro::logger::info("Failed to get device IDs");
        free(devices);
        return 1;
    }

    // Set default device
    Piro::logger::info("Active device");
    device = devices[DP.getvalue<int>(DeviceParams::ID)];
    Piro::print_device_info(device);

    // Initialize OpenCL
    Piro::logger::info("Calling OpenCL");
    // Create OpenCL context
    kernels.setvalue(Piro::kernels::CONTEXT, clCreateContext(NULL, 1, &device, NULL, NULL, &err));
    if (err != CL_SUCCESS) {
        Piro::logger::info("Failed to create OpenCL context");
        return 1;
    }

    // Create command queue
    kernels.setvalue(Piro::kernels::QUEUE, clCreateCommandQueue(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), device, 0, &err));
    if (err != CL_SUCCESS) {
        Piro::logger::info("Failed to create command queue");
        clReleaseContext(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT));
        return 1;
    }
    
    return 0;
}

cl_program Piro::opencl_CreateProgram(const char* dialog){
    Piro::kernels& kernels = Piro::kernels::getInstance();
    kernels.setvalue(Piro::kernels::PROGRAM, 
                    clCreateProgramWithSource(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), 1, 
                    &dialog, NULL, &err));
    if (err != CL_SUCCESS) {
        Piro::logger::info("Failed to create program with source");
        clReleaseCommandQueue(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
        clReleaseContext(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT));
    }
    return kernels.getvalue<cl_program>(Piro::kernels::PROGRAM);
}

cl_int Piro::opencl_BuildProgram(cl_program program){
    Piro::kernels& kernels = Piro::kernels::getInstance();
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
        clReleaseCommandQueue(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
        clReleaseContext(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT));
        return 1;
    }
    return err;
}

std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

// Function to save binary to file
void saveBinary(const std::string& filename, const std::vector<unsigned char>& binary) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    file.write(reinterpret_cast<const char*>(binary.data()), binary.size());
}

int Piro::loadKernelsFromFile(const std::string& filename, 
                                std::vector<std::string>& knm, 
                                std::vector<std::string>& ksm) {
    std::ifstream file(filename);
    if (!file.is_open()) {
    throw std::runtime_error("Failed to open kernel names file.");
    }
    int linecount = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;  // Skip empty lines

        // Kernel name is read from the file
        knm.push_back(line.c_str());

        // Load the corresponding kernel source from a separate file
        std::string kernelSource = Piro::readFile(line);
        ksm.push_back(kernelSource);
        linecount++;
    }

    file.close();
    return linecount;
}

int Piro::opencl_build(){
    // Create program objects
    int kernelmathcount, kernelcount; 
    Piro::logger::info("Building program : initiated");
    Piro::kernels& kernels = Piro::kernels::getInstance();
    Piro::logger::info("Reading kernel files from : ", current_path.string(), "/assets/kernels/0_kernels_math.txt");
    kernelmathcount = Piro::loadKernelsFromFile(current_path.string() + "/assets/kernels/0_kernels_math.txt",
                             kernelNamesmath, kernelSourcesmath);
    kernelcount = Piro::loadKernelsFromFile(current_path.string() + "/assets/kernels/0_kernels.txt",
                                kernelNames, kernelSources);
    std::vector<cl_program> programs_math(kernelmathcount);
    std::vector<cl_program> programs(kernelcount);
    std::vector<cl_kernel> kernel_math(kernelmathcount);
    std::vector<cl_kernel> kernel(kernelcount);
    kernels.setvalue(Piro::kernels::PROGRAM_MATH, programs_math);
    kernels.setvalue(Piro::kernels::PROGRAM, programs);
    kernels.setvalue(Piro::kernels::KERNEL_MATH, kernel_math);
    kernels.setvalue(Piro::kernels::KERNEL, kernel);
    // read the number of lines
    for(size_t i = 0; i < kernelmathcount; i++){
        const char* kernelSourcePtr = kernelSourcesmath[i].c_str();
        size_t kernelSourceSize = kernelSourcesmath[i].size();

        kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM_MATH)[i] = clCreateProgramWithSource(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), 1, &kernelSourcePtr, &kernelSourceSize, &err);
        err = Piro::opencl_BuildProgram(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM_MATH)[i]);
        kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[i] = clCreateKernel(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM_MATH)[i],
                                                                                    kernelNamesmath[i].c_str(), &err);

        // Export the compiled program binary
        size_t binarySize;
        err = clGetProgramInfo(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM_MATH)[i], CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, NULL);
        std::vector<unsigned char> binary(binarySize);
        unsigned char* binaryPtr = binary.data();
        err = clGetProgramInfo(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM_MATH)[i], CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binaryPtr, NULL);
        saveBinary(current_path.string() + "/assets/kernels/" + kernelNamesmath[i] + "_program.bin", binary);
        Piro::logger::info("Binary successfully saved to: ", kernelNamesmath[i]);
        Piro::logger::info("Binary size: " , binarySize , " bytes");
    }

    for(size_t i = 0; i < kernelcount; i++){

        const char* kernelSourcePtr = kernelSources[i].c_str();
        size_t kernelSourceSize = kernelSources[i].size();

        kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM)[i] = clCreateProgramWithSource(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), 1, &kernelSourcePtr, &kernelSourceSize, &err);
        err = Piro::opencl_BuildProgram(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM)[i]);
        kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[i] = clCreateKernel(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM)[i], kernelNames[i].c_str(), &err);

        // Export the compiled program binary
        size_t binarySize;
        err = clGetProgramInfo(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM)[i], CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, NULL);
        std::vector<unsigned char> binary(binarySize);
        unsigned char* binaryPtr = binary.data();
        err = clGetProgramInfo(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM)[i], CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binaryPtr, NULL);
        saveBinary(current_path.string() + "/assets/kernels/" + kernelNames[i] + "_program.bin", binary);
        Piro::logger::info("Binary successfully saved to: ", kernelNames[i]);
        Piro::logger::info("Binary size: " , binarySize , " bytes");
    }

    if (err != CL_SUCCESS){
        Piro::logger::info("program error");
    }
    Piro::logger::info("Building program : completed");
    return 0;
}


// Function to read binary file
std::vector<unsigned char> Piro::readBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open binary file: " + filename);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<unsigned char> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read binary file: " + filename);
    }
    
    return buffer;
}

int Piro::opencl_run(){
    int kernelmathcount, kernelcount;
    // Create program objects
    Piro::logger::info("Reading program : initiated");
    Piro::kernels& kernels = Piro::kernels::getInstance();
    Piro::logger::info("Reading kernel files from : ", current_path.string(), "/assets/kernels/0_kernels_math.txt");
    kernelmathcount = Piro::loadKernelsFromFile(current_path.string() + "/assets/kernels/0_kernels_math.txt",
                             kernelNamesmath, kernelSourcesmath);
    kernelcount = Piro::loadKernelsFromFile(current_path.string() + "/assets/kernels/0_kernels.txt",
                                kernelNames, kernelSources);
    std::vector<cl_program> programs_math(kernelmathcount);
    std::vector<cl_program> programs(kernelcount);
    std::vector<cl_kernel> kernel_math(kernelmathcount);
    std::vector<cl_kernel> kernel(kernelcount);
    kernels.setvalue(Piro::kernels::PROGRAM_MATH, programs_math);
    kernels.setvalue(Piro::kernels::PROGRAM, programs);
    kernels.setvalue(Piro::kernels::KERNEL_MATH, kernel_math);
    kernels.setvalue(Piro::kernels::KERNEL, kernel);

    for(size_t i = 0; i < kernelmathcount; i++){
        // Read binary from the file
        Piro::logger::info("Kernel names : ", kernelNamesmath[i], ", reading : " , current_path.string() + "/assets/kernels/" + kernelNamesmath[i] + "_program.bin");
        std::vector<unsigned char> binary = readBinaryFile(current_path.string() + "/assets/kernels/" + kernelNamesmath[i] + "_program.bin");
        
        const unsigned char* binaryPtr = binary.data();
        size_t binarySize = binary.size();
        cl_int binaryStatus;
        kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM_MATH)[i]  = clCreateProgramWithBinary(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), 1, &device, &binarySize,
                                                     &binaryPtr, &binaryStatus, &err);
        err = clBuildProgram(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM_MATH)[i], 1, &device, NULL, NULL, NULL);
        kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[i] = clCreateKernel(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM_MATH)[i], kernelNamesmath[i].c_str(), &err);
    }


    for(size_t i = 0; i < kernelcount; i++){
        // Read binary from the file
        Piro::logger::info("Kernel names : ", kernelNames[i], ", reading : " , current_path.string() + "/assets/kernels/" + kernelNames[i] + "_program.bin");
        std::vector<unsigned char> binary = readBinaryFile(current_path.string() + "/assets/kernels/" + kernelNames[i] + "_program.bin");
        // Create program from binary
        const unsigned char* binaryPtr = binary.data();
        size_t binarySize = binary.size();
        cl_int binaryStatus;
        kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM)[i]  = clCreateProgramWithBinary(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), 1, &device, &binarySize,
                                                     &binaryPtr, &binaryStatus, &err);
        if (err != CL_SUCCESS) {
            Piro::logger::info("Failed to create program with binary", err);
        }
        err = clBuildProgram(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM)[i], 1, &device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            // If loading failed, print build log
            size_t logSize;
            clGetProgramBuildInfo(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM)[i], device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
            std::vector<char> buildLog(logSize);
            clGetProgramBuildInfo(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM)[i], device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), NULL);
            std::cerr << "Build Error: " << buildLog.data() << std::endl;
            throw std::runtime_error("Failed to build program");
        }
        kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[i] = clCreateKernel(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM)[i], kernelNames[i].c_str(), &err);
        if (err != CL_SUCCESS) {
            Piro::logger::info("Failed to create program with binary", err);
        }
    }

    
    Piro::logger::info("Running program : completed");
    return 0;

}

int Piro::opencl_cleanup(){
    
    Piro::logger::info("Cleanup started");
    Piro::kernels& kernels = Piro::kernels::getInstance();
    // Clean up
    free(devices);
    // Cleanup
    clReleaseMemObject(memBx);
    clReleaseMemObject(memCx);
    clReleaseMemObject(memDx);
    clReleaseMemObject(memEx);

    for(size_t i = 0; i < kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM_MATH).size(); i++){
        clReleaseKernel(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[i]);
        clReleaseProgram(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM_MATH)[i]);
    }
    for(size_t i = 0; i < kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM).size(); i++){
        clReleaseKernel(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[i]);
        clReleaseProgram(kernels.getvalue<std::vector<cl_program>>(Piro::kernels::PROGRAM)[i]);
    }
    clReleaseCommandQueue(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
    clReleaseContext(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT));

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