#ifndef openclutilities_hpp
#define openclutilities_hpp

#include <vector>
#include <CL/opencl.h>

namespace Piro::opencl_utilities{
    template <typename T>
    std::vector<T> copyCL(cl_command_queue queue, cl_mem memC, int N, cl_event *event6);

    template <typename U>
    int copyCL_offset(cl_command_queue queue, cl_mem memC, std::vector<U>& Lap, int offset, int N, cl_event *event6);
}
#endif