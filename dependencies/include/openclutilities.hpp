#ifndef openclutilities_hpp
#define openclutilities_hpp
#include <vector>
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#elif _WIN32
    #include <CL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

namespace Piro::opencl_utilities{
    template <typename T>
    std::vector<T> copyCL(cl_command_queue queue, cl_mem memC, int N, cl_event *event6);

    template <typename U>
    int copyCL_offset(cl_command_queue queue, cl_mem memC, std::vector<U>& Lap, int offset, int N, cl_event *event6);
}
#endif