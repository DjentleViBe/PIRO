#ifndef bc_hpp
#define bc_hpp

#include <vector>
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#elif _WIN32
    #include "./CL/opencl.h"
#else
    #include "./CL/opencl.h"
#endif

extern std::vector<std::vector<int>> indices;
extern std::vector<int> indices_toprint;
extern std::vector<int> indices_toprint_vec;
extern cl_int err;
void readbc();
void setbc();
void opencl_setBC(int ind);

#endif