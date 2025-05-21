__kernel void SpMACSR(__global float* values_A,
                        __global const int* columns_A,
                        __global const int* rowPointers_A,

                        __global float* values_B,
                        __global const int* columns_B,
                        __global const int* rowPointers_B,

                        __global float* values_C,
                        __global const int* columns_C,
                        __global const int* rowPointers_C,

                        __global float* values_D,
                        __global int* columns_D,
                        __global int* rowPointers_D,
                        
                        const int num_rows,
                        const int num_cols){
    int row = get_global_id(0);
    
    if (row >= num_rows) return;
    
    // Get row ranges for each matrix
    int a_start = rowPointers_A[row];
    int a_end = rowPointers_A[row + 1];
    
    int b_start = rowPointers_B[row];
    int b_end = rowPointers_B[row + 1];
    
    int c_start = rowPointers_C[row];
    int c_end = rowPointers_C[row + 1];
    
    // Initialize output row pointer
    int r_start = rowPointers_D[row];
    int r_idx = r_start;
    
    // Merge the three rows
    int a_idx = a_start;
    int b_idx = b_start;
    int c_idx = c_start;
    
    while (a_idx < a_end || b_idx < b_end || c_idx < c_end) {
        int min_col = num_cols;
        
        // Find the minimum column index among the three matrices
        if (a_idx < a_end) min_col = min(min_col, columns_A[a_idx]);
        if (b_idx < b_end) min_col = min(min_col, columns_B[b_idx]);
        if (c_idx < c_end) min_col = min(min_col, columns_C[c_idx]);
        
        // Sum values from all matrices at this column
        float sum = 0.0f;
        
        if (a_idx < a_end && columns_A[a_idx] == min_col) {
            sum += values_A[a_idx++];
        }
        
        if (b_idx < b_end && columns_B[b_idx] == min_col) {
            sum += values_B[b_idx++];
        }
        
        if (c_idx < c_end && columns_C[c_idx] == min_col) {
            sum += values_C[c_idx++];
        }
        
        // Store the result if non-zero
        if (fabs(sum) > 1e-7f) {
            columns_D[r_idx] = min_col;
            values_D[r_idx] = sum;
            r_idx++;
        }
        rowPointers_D[row + 1] = r_idx;
    }
}
