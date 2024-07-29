#ifndef KERNEL_CL_H
#define KERNEL_CL_H

const char *kernelSource = R"CLC(
// Define your OpenCL kernel code here
__kernel void matrixMultiply(__global const float *A,
                        __global const float *B,
                        __global float *C,
                        uint N, uint M, uint P) {
    // Get the index of the current element to be processed
    uint row = get_global_id(1);
    uint col = get_global_id(0);

    // Perform the vector addition
   
    if (row < N && col < P) {
        float sum = 0.0f;
        for(uint k = 0; k < M; k++){
            sum += A[row * M + k] * B[k * P + col];
        }
        C[row * P + col] = sum;

    }
    
}
)CLC";


// Kernel source for boundary conditions
const char *setBC = R"CLC(
__kernel void setBC(__global float *A,
                    __global const float *B,
                    __global uint *indices,
                    uint size) {
    uint id = get_global_id(0);
    if (id < size) {
        A[indices[id]] = B[indices[id]];
    }
}
)CLC";

const char *laplaciancalc = R"CLC(
__kernel void laplacian(__global float *B,
                        __global float *C,
                        float delta_x,
                        float delta_y,
                        float delta_z,
                        uint nx, uint ny,
                        float deltat,
                        uint size) {
    uint id = get_global_id(0);
    if (id < size) {
        uint z = id / (nx * ny);
        uint y = (id / nx) % ny;
        uint x = id % nx;

        if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < (size / (nx * ny)) - 1) {
                C[id] = deltat * ((B[id + 1] + B[id - 1] - 2.0f * B[id]) / (delta_x * delta_x) 
                      + (B[id + nx] + B[id - nx] - 2.0f * B[id]) / (delta_y * delta_y)
                      + (B[id + nx * ny] + B[id - nx * ny] - 2.0f * B[id]) / (delta_z * delta_z));
            }
        }
}
)CLC";

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

const char *subtractVectors = R"CLC(
__kernel void addVectors(__global float *A,
                        __global const float *B,
                        __global const float *C,
                        uint size) {
    uint id = get_global_id(0);
    if (id < size) {
        A[id] = B[id] - C[id];
    }
}
)CLC";

const char *divideVectors = R"CLC(
__kernel void addVectors(__global float *A,
                        __global const float *B,
                        __global const float *C,
                        uint size) {
    uint id = get_global_id(0);
    if (id < size) {
        A[id] = B[id] - C[id];
    }
}
)CLC";

#endif // KERNEL_CL_H