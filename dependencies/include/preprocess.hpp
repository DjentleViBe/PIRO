#ifndef preprocess_hpp
#define preprocess_hpp

#include "datatypes.hpp"
#include <vector>
#include <string>
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#elif _WIN32
    #include "./CL/opencl.h"
#else
    #include "./CL/opencl.h"
#endif

extern Giro::MeshParams MP;
extern Giro::SolveParams SP;
extern Giro::DeviceParams DP;
extern int ts;

extern std::vector<cl_program> program_math;
extern std::vector<cl_kernel> kernel_math;

extern cl_program   program_gradient, 
                    program_laplacian,
                    program_setBC;
extern cl_kernel    kernelgradient,
                    kernellaplacian,
                    kernelBC;

extern cl_context   context;
extern cl_command_queue queue;
int preprocess(const std::string& name);
int idx(int i, int j, int k, int N_x, int N_y);
int opencl_init();
int opencl_build();

#endif