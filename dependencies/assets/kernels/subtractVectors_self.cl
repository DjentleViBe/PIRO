__kernel void subtractVectors_self(__global float *A,
                            __global const float *B,
                            uint size) {
        uint id = get_global_id(0);
        if (id < size) {
            A[id] = A[id] - B[id];
        }
    }