#ifndef openclutilities_hpp
#define openclutilities_hpp

#include <vector>
#include <CL/opencl.h>

namespace Piro::kernels{
    extern std::vector<cl_program> program_math;
    extern std::vector<cl_kernel> kernel_math;
    extern std::vector<cl_program> program;
    extern std::vector<cl_kernel> kernel;
    extern cl_context   context;
    extern cl_command_queue queue;
    extern cl_uint maxWorkGroupSize;
}

namespace Piro::opencl_utilities{
    template <typename T>
    std::vector<T> copyCL(cl_command_queue queue, cl_mem memC, int N, cl_event *event6);

    template <typename U>
    int copyCL_offset(cl_command_queue queue, cl_mem memC, std::vector<U>& Lap, int offset, int N, cl_event *event6);
}

extern int compile;
#endif