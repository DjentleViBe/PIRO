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
        
        // Ensure all threads complete before the next iteration
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}