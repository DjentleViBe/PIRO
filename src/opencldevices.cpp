#define CL_HPP_TARGET_OPENCL_VERSION 200 
#include <iostream>
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#elif _WIN32
    #include "../dependencies/include/CL/opencl.h"
#else
    #include "../dependencies/include/CL/opencl.h"
#endif

static void print_device_info(cl_device_id device){
    char name[128];
    char vendor[128];

    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    std::cout << name << ", ";
    std::cout << vendor << std::endl;

}

int main(){
     // Initialize OpenCL
    cl_uint platformCount;
    cl_uint num_devices;
    cl_device_id *devices;
    cl_device_id device;
    clGetPlatformIDs(0, nullptr, &platformCount);
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
   
    int j = 0;
    for (cl_platform_id platform : platforms) {
        size_t size;
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &size);
        std::string name(size - 1, '\0');
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, &name[0], nullptr);
        std::cout << "Platform " << j << " : " << name << std::endl;

        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 3, NULL, &num_devices);
        devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
        clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        for (cl_uint i = 0; i < num_devices; ++i) {
            std::cout << "  Device " << i << " : ";
            device = devices[i];
            print_device_info(device);
        }
        j++;
    }
    return 0;
}