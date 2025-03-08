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

const char *sparseMatrixMultiplyCSR = R"CLC(
__kernel void sparseMatrixMultiplyCSR(const int M,
                                        const int K,
                                        const int R,
                                        __global const float* values_A,
                                        __global const int* columns_A,
                                        __global const int* rowPointers_A,
                                        float deltat,
                                        __global float* B,
                                        __global float* C) {
    int row = get_global_id(0); // 1D index for row

    //if (row >= M) return; // Ensure we don't go out of bounds

    // Iterate over all columns of C
    for (int col = 0; col < R; ++col) {
        float result = 0.0f;

        // Iterate over the non-zero entries in row `row` of matrix A
        for (int i = rowPointers_A[row]; i < rowPointers_A[row + 1]; ++i) {
            int col_A = columns_A[i];  // Column index of the non-zero value in A
            float value_A = values_A[i];  // Value at position (row, col_A) in matrix A

            // Multiply with the corresponding value from matrix B
            result += value_A * B[col_A * R + col];  // B[col_A, col]
        }

        // Store the result in the output matrix C
        C[row * R + col] = deltat * result;
        printf("C[%u] = %f\n", row * R + col, C[row * R + col]);
    }
}
)CLC";

const char *LaplaciansparseMatrixMultiplyCSR = R"CLC(
    __kernel void LaplaciansparseMatrixMultiplyCSR(const int M,
                                            const int K,
                                            const int R,
                                            __global const float* values_A,
                                            __global const int* columns_A,
                                            __global const int* rowPointers_A,
                                            float deltat,
                                            __global float* B,
                                            __global float* C,
                                            uint nx, uint ny,
                                            uint size) {
        int row = get_global_id(0); // 1D index for row
    
        if (row >= M) return; // Ensure we don't go out of bounds
    
        // Iterate over all columns of C
        for (int col = 0; col < R; ++col) {
            float result = 0.0f;
    
            // Iterate over the non-zero entries in row `row` of matrix A
            for (int i = rowPointers_A[row]; i < rowPointers_A[row + 1]; ++i) {
                int col_A = columns_A[i];  // Column index of the non-zero value in A
                float value_A = values_A[i];  // Value at position (row, col_A) in matrix A
    
                // Multiply with the corresponding value from matrix B
                result += value_A * B[col_A * R + col];  // B[col_A, col]
                // printf("B[%u] = %f\n", col_A * R + col, B[col_A * R + col]);
            }
            // Store the result in the output matrix C
            int id = row * R + col;
            uint z = id / (nx * ny);
            uint y = (id / nx) % ny;
            uint x = id % nx;
            if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < (size / (nx * ny)) - 1) {
                C[row * R + col] = deltat * result;
                // printf("C[%u] = %f\n", row * R + col, C[row * R + col]);
            }
            
        }
    }
    )CLC";

const char *laplacianscalar = R"CLC(
__kernel void laplacianscalar(__global float *B,
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
                printf("C[%u] = %f\n", id, C[id]);
                printf("B[%u] = %f\n", id, B[id]);
                printf("B[%u] = %f\n", id + 1, B[id + 1]);
                printf("B[%u] = %f\n", id - 1, B[id - 1]);

                printf("B[%u] = %f\n", id + nx, B[id + nx]);
                printf("B[%u] = %f\n", id - nx, B[id - nx]);
                printf("B[%u] = %f\n",id + nx * ny, B[id + nx * ny]);
                printf("B[%u] = %f\n",id - nx * ny, B[id - nx * ny]);
            }
        }
}
)CLC";

const char *laplacianvector = R"CLC(
__kernel void laplacianvector(__global float *B,
                        __global float *C,
                        float delta_x,
                        float delta_y,
                        float delta_z,
                        uint nx, uint ny,
                        float deltat,
                        uint size) {
    uint id = get_global_id(0);
    uint z = id / (nx * ny);
    uint y = (id / nx) % ny;
    uint x = id % nx;
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < (size / (nx * ny)) - 1) {
        if (id < size) {
            C[id] = deltat * ((B[id + 1] + B[id - 1] - 2.0f * B[id]) / (delta_x * delta_x) 
                      + (B[id + nx] + B[id - nx] - 2.0f * B[id]) / (delta_y * delta_y)
                      + (B[id + nx * ny] + B[id - nx * ny] - 2.0f * B[id]) / (delta_z * delta_z));
            }
        else if (id > size && id < 2 * size) {
            C[id] = deltat * ((B[id + 1] + B[id - 1] - 2.0f * B[id]) / (delta_x * delta_x) 
                      + (B[id + nx] + B[id - nx] - 2.0f * B[id]) / (delta_y * delta_y)
                      + (B[id + nx * ny] + B[id - nx * ny] - 2.0f * B[id]) / (delta_z * delta_z));
            }
        
        else {
            C[id] = deltat * ((B[id + 1] + B[id - 1] - 2.0f * B[id]) / (delta_x * delta_x) 
                      + (B[id + nx] + B[id - nx] - 2.0f * B[id]) / (delta_y * delta_y)
                      + (B[id + nx * ny] + B[id - nx * ny] - 2.0f * B[id]) / (delta_z * delta_z));
            }
        
        }
}
)CLC";

const char *gradcalc1 = R"CLC(
__kernel void gradient1(__global float *B,
                        __global float *C,
                        __global float *ind,
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
            C[id] = deltat * (ind[id] * ((B[id + 1] - B[id - 1]) / (2 * delta_x))
                    + ind[size + id] * ((B[id + nx] - B[id - nx]) / (2 * delta_y))
                    + ind[size * 2 + id] * ((B[id + nx * ny] - B[id - nx * ny]) / (2 * delta_z)));
        
            }
        }
}
)CLC";

const char *gradcalc2 = R"CLC(
__kernel void gradient2(__global float *B,
                        __global float *C,
                        __global float *ind,
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
            C[id] = deltat * (ind[id] * ((B[id + 1] - B[id - 1]) / (2 * delta_x))
                    + ind[size + id] * ((B[id + nx] - B[id - nx]) / (2 * delta_y))
                    + ind[size * 2 + id] * ((B[id + nx * ny] - B[id - nx * ny]) / (2 * delta_z)));
        
            }
        }
}
)CLC";

const char *gradcalc3 = R"CLC(
__kernel void gradient3(__global float *B,
                        __global float *C,
                        __global float *ind,
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
            C[id] = deltat * (ind[id] * ((B[id + 1] - B[id - 1]) / (2 * delta_x))
                    + ind[size + id] * ((B[id + nx] - B[id - nx]) / (2 * delta_y))
                    + ind[size * 2 + id] * ((B[id + nx * ny] - B[id - nx * ny]) / (2 * delta_z)));
        
            }
        }
}
)CLC";

const char *gradcalc4 = R"CLC(
__kernel void gradient4(__global float *B,
                        __global float *C,
                        __global float *D,
                        float delta_x,
                        float delta_y,
                        float delta_z,
                        uint nx, uint ny,
                        float deltat,
                        uint size) {
    uint id = get_global_id(0);
    uint z = id / (nx * ny);
    uint y = (id / nx) % ny;
    uint x = id % nx;
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < (size / (nx * ny)) - 1) {
        if (id < size) {
            C[id] = deltat *    ((D[id + 1] - D[id - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_x) + 
                                (D[size + id + 1] - D[size + id - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_y) + 
                                (D[size * 2 + id + 1] - D[size * 2 + id - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_z));
        }
        else if(id > size && id < 2 * size){
            C[id] = deltat *    ((D[id - size + 1] - D[id - size - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_x) + 
                                (D[id + 1] - D[id - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_y) + 
                                (D[size + id + 1] - D[size + id - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_z));
        
        }
        else{
            C[id] = deltat *    ((D[id - 2 * size + 1] - D[id - 2 * size - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_x) + 
                                (D[id - size + 1] - D[id - size - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_y) + 
                                (D[id + 1] - D[id - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_z));
        
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

const char *divideVectors = R"CLC(
__kernel void divideVectors(__global float *A,
                        __global const float *B,
                        __global const float *C,
                        uint size) {
    uint id = get_global_id(0);
    if (id < size) {
        A[id] = B[id] / C[id];
    }
}
)CLC";

const char *divideVectors_constant = R"CLC(
__kernel void divideVectors_constant(__global float *A,
                                float B,
                                __global const float *C,
                                uint size) {
    uint id = get_global_id(0);
    if (id < size) {
        A[id] = C[id] / B;
    }
}
)CLC";


#endif // KERNEL_CL_H