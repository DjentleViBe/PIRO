#ifndef preprocess_hpp
#define preprocess_hpp

#include "datatypes.hpp"
#include <vector>
#include <string>
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#elif _WIN32
    #include "../dependencies/include/CL/opencl.h"
#else
    #include "../dependencies/include/CL/opencl.h"
#endif

extern Giro::MeshParams MP;
extern Giro::SolveParams SP;
extern Giro::DeviceParams DP;
extern std::vector<std::vector<float>> scagradmatrix, scadivmatrix;
extern int ts;
extern float* scalapvectorpointer;

extern cl_program   program_multiplyVec, 
                    program_laplacian, 
                    program_setBC;
extern cl_kernel    kernel_multiplyVec,
                    kernellaplacian,
                    kernelBC;
extern cl_context   context;
extern cl_command_queue queue;
int preprocess(const std::string& name);
int idx(int i, int j, int k, int N_x, int N_y);
int opencl_init();
int opencl_build();

#endif