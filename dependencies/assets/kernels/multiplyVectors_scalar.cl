__kernel void multiplyVectors_scalar(__global const float *B,
                                    __global float *C,
                                    uint size) {
        uint id = get_global_id(0);
        if (id < size) {
            C[id] = B[id % (size / 3)] * C[id];
        }
    }