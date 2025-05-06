#ifndef DIVIDE_VECTORS_CL_H
#define DIVIDE_VECTORS_CL_H

const char *divideVectors = R"CLC(
__kernel void divideVectors(__global float *A,
                        __global const float *B,
                        __global const float *C,
                        uint size) {
    uint id = get_global_id(0);
    if (id < size) {
        A[id] = B[id] / C[id];
    }
}
)CLC";
    
const char *divideVectors_constant = R"CLC(
__kernel void divideVectors_constant(__global float *A,
                                float B,
                                __global const float *C,
                                uint size) {
    uint id = get_global_id(0);
    if (id < size) {
        A[id] = C[id] / B;
    }
}
)CLC";

#endif 