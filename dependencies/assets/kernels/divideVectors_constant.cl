__kernel void divideVectors_constant(__global float *A,
                                float B,
                                __global const float *C,
                                uint size) {
    uint id = get_global_id(0);
    if (id < size) {
        A[id] = C[id] / B;
    }
}