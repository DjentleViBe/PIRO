__kernel void SpMVCSR_I1(const int M,
                                            const int K,
                                            const int R,
                                            __global const float* values_A,
                                            __global const int* columns_A,
                                            __global const int* rowPointers_A,
                                            const float deltat,
                                            __global float* B,
                                            __global float* C,
                                            uint nx, uint ny,
                                            uint size) {
        int row = get_global_id(0); // 1D index for row
    
        if (row > M) return; // Ensure we don't go out of bounds
    
        // Iterate over all columns of C
        for (int col = 0; col < R; ++col) {
            float result = 0.0f;
    
            // Iterate over the non-zero entries in row `row` of matrix A
            for (int i = rowPointers_A[row]; i < rowPointers_A[row + 1]; ++i) {
                int col_A = columns_A[i];  // Column index of the non-zero value in A
                float value_A = values_A[i];  // Value at position (row, col_A) in matrix A
                result += deltat * value_A * B[col_A * R + col];
            }
            // Store the result in the output matrix C
            int id = row * R + col;
            uint z = id / (nx * ny);
            uint y = (id / nx) % ny;
            uint x = id % nx;
            if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < (size / (nx * ny)) - 1) {
                C[row * R + col] = B[row * R + col] - result;
                // printf("C[%u] = %f\n", row * R + col, C[row * R + col]);
            }
            
        }
    }