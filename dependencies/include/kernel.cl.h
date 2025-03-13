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
        // printf("C[%u] = %f\n", row * R + col, C[row * R + col]);
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

const char *lu_decomposition = R"CLC(
    __kernel void lu_decomposition(
        __global const int* row_ptr,     // CSR values of A
        __global const int* col_idx,      // CSR row pointers
        __global const float* values,      // CSR column indices
        __global int* L_row_ptr,          // CSR row pointers for L
        __global int* L_col_idx,          // CSR column indices for L
        __global float* L_values,         // CSR values of L
        __global int* U_row_ptr,          // CSR row pointers for U
        __global int* U_col_idx,          // CSR column indices for U
        __global float* U_values,         // CSR values of U
        int N                              // Matrix size
    ) {
    
    int row = get_global_id(0);
    if (row >= N) return; // Out of bounds check

    // Perform the decomposition: L = lower triangular, U = upper triangular
    // Initialize L and U as identity matrix (this is a simplification)

    for (int k = row_ptr[row]; k < row_ptr[row + 1]; k++) {
        int col = col_idx[k];
        if (row == col) {
            // Diagonal element for U
            U_values[k] = values[k];
            L_values[k] = 1.0f;
        } else if (row < col) {
            // Upper triangular part (U)
            U_values[k] = values[k];
            L_values[k] = 0.0f;
        } else {
            // Lower triangular part (L)
            L_values[k] = values[k];
            U_values[k] = 0.0f;
        }
    }
    }
    )CLC";
    

const char *forward_substitution_csr = R"CLC(
__kernel void forward_substitution_csr(
    __global const int* L_row_ptr,
    __global const int* L_col_idx,
    __global const float* L_values,
    __global float* y,
    __global const float* b,
    int N
) {
    int i = get_global_id(0);

    if (i < N) {
        float sum = b[i];
        for (int j = L_row_ptr[i]; j < L_row_ptr[i + 1]; j++) {
            int col = L_col_idx[j];
            if (col < i) {
                sum -= L_values[j] * y[col];
            }
        }
        y[i] = sum;  // L is lower triangular, diagonal assumed to be 1
    }
}
)CLC";

const char *backward_substitution_csr = R"CLC(
__kernel void backward_substitution_csr(
    __global const int* U_row_ptr,
    __global const int* U_col_idx,
    __global const float* U_values,
    __global float* x,
    __global const float* y,
    int N
) {
    int i = get_global_id(0);

    if (i < N) {
        int row = N - 1 - i; // Process from last row to first
        float sum = y[row];

        for (int j = U_row_ptr[row]; j < U_row_ptr[row + 1]; j++) {
            int col = U_col_idx[j];
            if (col > row) {
                sum -= U_values[j] * x[col];
            } else if (col == row) {
                x[row] = sum / U_values[j];
            }
        }
    }
}
)CLC";
#endif // KERNEL_CL_H