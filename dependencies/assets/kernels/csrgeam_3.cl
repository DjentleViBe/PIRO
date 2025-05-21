__kernel void csrgeam_3(__global const float* values_A,
                        __global const int* columns_A,
                        __global const int* rowPointers_A,

                        __global const float* values_B,
                        __global const int* columns_B,
                        __global const int* rowPointers_B,
                        
                        __global const float* values_C,
                        __global const int* columns_C,
                        __global const int* rowPointers_C,

                        __global float* values_D,
                        __global int* columns_D,
                        __global int* rowPointers_D,

                        __global int* phase_one,
                        __global const float* B,
                        const int num_rows){
    
    int row = get_global_id(0);
    if (row > num_rows) return;

    int a_start = rowPointers_A[row];
    int a_end = rowPointers_A[row + 1];
    
    int b_start = rowPointers_B[row];
    int b_end = rowPointers_B[row + 1];

    int c_start = rowPointers_C[row];
    int c_end = rowPointers_C[row + 1];

    int d_index = rowPointers_D[row];
    int i = a_start;
    int j = b_start;
    int k = c_start;
    while(i < a_end && j < b_end && k < c_end){
        
        int colA = columns_A[i];
        int colB = columns_B[j];
        int colC = columns_C[k];
        int min_col = min(colA, min(colC, colB));
        if(colA == min_col){
            // printf("values[%d] = %f\n", i, values_A[i]);
            columns_D[d_index] = colA;
            values_D[d_index]  = values_A[i] * B[row];
            i++;
        }
        else if(colB == min_col){
            // printf("values[%d] = %f\n", i, values_B[i]);
            columns_D[d_index] = colB;
            values_D[d_index]  = values_B[j] * B[num_rows + row];
            j++;
        }
        else if(colC == min_col){
            // printf("values[%d] = %f\n", i, values_B[i]);
            columns_D[d_index] = colC;
            values_D[d_index]  = values_C[k] * B[num_rows * 2 + row];
            k++;
        }
        else{
            columns_D[d_index] = colA;  // colA == colB == colC
            values_D[d_index]  = values_A[i] * B[row] + values_B[j] * B[num_rows + row] + values_C[k] * B[num_rows * 2 + row];
            i++;
            j++;
            k++;
        }
        d_index++;
    }

    while(i < a_end){
        columns_D[d_index] = columns_A[i];
        values_D[d_index]  = values_A[i] * B[row];
        i++;
        d_index++;
    }
    while(j < b_end){
        columns_D[d_index] = columns_B[j];
        values_D[d_index]  = values_B[j] * B[num_rows + row];
        j++;
        d_index++;
    }
    while(k < c_end){
        columns_D[d_index] = columns_C[k];
        values_D[d_index]  = values_C[k] * B[num_rows * 2 + row];
        k++;
        d_index++;
    }
}