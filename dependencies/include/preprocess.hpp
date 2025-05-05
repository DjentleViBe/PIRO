#ifndef preprocess_hpp
#define preprocess_hpp

#include <datatypes.hpp>
#include <vector>
#include <string>
#include <CL/opencl.h>

extern Piro::MeshParams MP;
extern Piro::SolveParams SP;
extern Piro::DeviceParams DP;
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

namespace Piro{
    int laplacian_CSR_init();
    int preprocess(const std::string& name);
    void print_device_info(cl_device_id device);
    cl_int opencl_BuildProgram(cl_program program);
    cl_program opencl_CreateProgram(const char* dialog);
    int opencl_init();
    int opencl_build();
}

#endif