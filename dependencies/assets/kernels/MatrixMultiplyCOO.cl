__kernel void matrixMultiplyCOO(const int M,
                                const int K,
                                const int nnz,
                                __global const float* values_A,
                                __global const int* columns_A,
                                __global const int* rows_A,
                                float deltat,
                                __global float* B,
                                __global float* C,
                                uint nx, uint ny,
                                uint size) {
        int row = get_global_id(0); // 1D index for row
    
        if (row >= M) return; // Ensure we don't go out of bounds
        float result = 0.0f;
        // Iterate over all columns of C
        for (int ind = 0; ind < nnz; ++ind) {
            
            if(rows_A[ind] == row){
                result += values_A[ind] * B[columns_A[ind]];
            }
            // Store the result in the output matrix C
        }
        C[row] = deltat * result;
    }