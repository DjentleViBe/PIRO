__kernel void subtractVectors(__global float *A,
                            __global const float *B,
                            __global const float *C,
                            uint size) {
        uint id = get_global_id(0);
        if (id < size) {
            A[id] = B[id] - C[id];
        }
    }