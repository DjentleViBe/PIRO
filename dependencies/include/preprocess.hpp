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
extern bool LAP_INIT;
extern bool RHS_INIT;

extern std::vector<cl_program> program_math;
extern std::vector<cl_kernel> kernel_math;

extern cl_program   program_gradient_scalar, 
                    program_gradient_vector, 
                    program_laplacian_scalar,
                    program_sparseMatrixMultiplyCSR,
                    program_lu_decompose_dense,
                    program_laplacian_vector,
                    program_setBC,
                    program_filter_array,
                    program_filter_row,
                    program_forward_substitution_csr,
                    program_backward_substitution_csr;
extern cl_kernel    kernelgradient_type1,
                    kernelgradient_type2,
                    kernelgradient_type3,
                    kernelgradient_type4,
                    kernellaplacianscalar,
                    kernellu_decompose_dense,
                    kernelsparseMatrixMultiplyCSR,
                    kernellaplaciansparseMatrixMultiplyCSR,
                    kernellaplacianvector,
                    kernelBC,
                    kernelfilterarray,
                    kernelfilterrow,
                    kernelforward_substitution_csr,
                    kernelbackward_substitution_csr;

extern cl_context   context;
extern cl_command_queue queue;
extern cl_uint maxWorkGroupSize;
int preprocess(const std::string& name);
int idx(int i, int j, int k, int N_x, int N_y);
int opencl_init();
int opencl_build();
int laplacian_CSR_init();
#endif