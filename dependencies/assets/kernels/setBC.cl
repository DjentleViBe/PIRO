__kernel void setBC(__global float *A,
                    __global const float *B,
                    __global uint *indices,
                    uint size) {
    uint id = get_global_id(0);
    if (id < size) {
        A[indices[id]] = B[indices[id]];
    }
}