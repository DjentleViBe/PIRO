#ifndef GPUINIT_HPP
#define GPUINIT_HPP

#include "datatypes.hpp"

extern float* B_ptr;
namespace Piro{
    int opencl_laplacian(float* hostB, int time, uint N, uint M, uint P);
}

#endif