#ifndef ADD_VECTORS_CL_H
#define ADD_VECTORS_CL_H

const char *addVectors = R"CLC(
    __kernel void addVectors(__global float *A,
                            __global const float *B,
                            __global const float *C,
                            uint size) {
        uint id = get_global_id(0);
        if (id < size) {
            A[id] = B[id] + C[id];
        }
    }
)CLC";

const char *addVectors_constant = R"CLC(
    __kernel void addVectors_constant(__global float *A,
                                    float B,
                                    __global const float *C,
                                    uint size) {
        uint id = get_global_id(0);
        if (id < size) {
            A[id] = B + C[id];
        }
    }
)CLC";

#endif 