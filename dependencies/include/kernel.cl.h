#ifndef KERNEL_CL_H
#define KERNEL_CL_H

const char *kernelSource = R"CLC(
// Define your OpenCL kernel code here
__kernel void matrixMultiply(__global const float *A,
                        __global float *B,
                        int N, int M, int P) {
    // Get the index of the current element to be processed
    int row = get_global_id(1);
    int col = get_global_id(0);

    // Perform the vector addition
   
    if (row < N && col < P) {
        float sum = 0.0f;
        for(int k = 0; k < M; k++){
            sum += A[row * M + k] * B[k * P + col];
        }
        B[row * P + col] = sum;

    }
    
}
)CLC";

#endif // KERNEL_CL_H