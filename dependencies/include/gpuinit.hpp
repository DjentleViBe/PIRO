#ifndef GPUINIT_HPP
#define GPUINIT_HPP

extern float* B_ptr;
int opencl_call(float* hostA, float* hostB, int time, int N, int M, int P);

#endif