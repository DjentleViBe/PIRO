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
            if (y[row] != 0.0f) { // Only compute if y[row] is non-zero
                float sum = y[row];
                // printf("y[%u] = %f\n", row, y[row]);
                // Compute summation term for backward substitution
                for (int j = U_row_ptr[row]; j < U_row_ptr[row + 1]; j++) {
                    
                    int col = U_col_idx[j];
                    if (col > row) {
                        sum -= U_values[j] * x[col];
                        // printf("x[%u] = %f\n", col, x[col]);
                    } else if (col == row) {
                        x[row] = sum / U_values[j]; // Store result when we reach the diagonal element
                        break;
                    }
                }
                
                // Debug output to check the sum and x values
                // printf("Row: %d, sum: %f, x[%d]: %f\n", row, sum, row, x[row]);
            }
        }
    }