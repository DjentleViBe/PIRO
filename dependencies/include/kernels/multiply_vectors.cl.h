#ifndef MULTIPLY_VECTORS_CL_H
#define MULTIPLY_VECTORS_CL_H

const char *multiplyVectors = R"CLC(
    __kernel void multiplyVectors(__global float *A,
                                    __global const float *B,
                                    __global const float *C,
                                    uint size) {
        uint id = get_global_id(0);
        if (id < size) {
            A[id] = B[id] * C[id];
        }
    }
)CLC";

const char *multiplyVectors_constant = R"CLC(
    __kernel void multiplyVectors_constant(__global float *A,
                                    float B,
                                    __global const float *C,
                                    uint size) {
        uint id = get_global_id(0);
        if (id < size) {
            A[id] = B * C[id];
        }
    }
)CLC";

#endif