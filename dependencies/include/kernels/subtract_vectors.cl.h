#ifndef SUBTRACT_VECTORS_CL_H
#define SUBTRACT_VECTORS_CL_H

const char *subtractVectors = R"CLC(
    __kernel void subtractVectors(__global float *A,
                            __global const float *B,
                            __global const float *C,
                            uint size) {
        uint id = get_global_id(0);
        if (id < size) {
            A[id] = B[id] - C[id];
        }
    }
)CLC";

const char *subtractVectors_constant = R"CLC(
    __kernel void subtractVectors_constant(__global float *A,
                                    float B,
                                    __global const float *C,
                                    uint size) {
        uint id = get_global_id(0);
        if (id < size) {
            A[id] = C[id] - B;
        }
    }
)CLC";

const char *subtractVectors_self = R"CLC(
    __kernel void subtractVectors_self(__global float *A,
                            __global const float *B,
                            uint size) {
        uint id = get_global_id(0);
        if (id < size) {
            A[id] = A[id] - B[id];
        }
    }
)CLC";

#endif