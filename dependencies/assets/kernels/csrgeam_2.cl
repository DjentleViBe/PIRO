__kernel void csrgeam_2(__global const float* values_A,
                        __global const int* columns_A,
                        __global const int* rowPointers_A,

                        __global const float* values_B,
                        __global const int* columns_B,
                        __global const int* rowPointers_B,
                        
                        __global float* values_C,
                        __global int* columns_C,
                        __global int* rowPointers_C,

                        __global int* phase_one,
                        const int num_rows){
    
    int row = get_global_id(0);
    if (row > num_rows) return;

    int a_start = rowPointers_A[row];
    int a_end = rowPointers_A[row + 1];
    
    int b_start = rowPointers_B[row];
    int b_end = rowPointers_B[row + 1];

    int c_index = rowPointers_C[row];
    int i = a_start;
    int j = b_start;

    while(i < a_end && j < b_end){
        
        int colA = columns_A[i];
        int colB = columns_B[j];
        if(colA < colB){
            // printf("values[%d] = %f\n", i, values_A[i]);
            columns_C[c_index] = colA;
            values_C[c_index]  = values_A[i];
            i++;
        }
        else if(colA > colB){
            // printf("values[%d] = %f\n", i, values_B[i]);
            columns_C[c_index] = colB;
            values_C[c_index]  = values_B[j];
            j++;
        }
        else{
            columns_C[c_index] = colA;  // colA == colB
            values_C[c_index]  = values_A[i] + values_B[j];
            i++;
            j++;
        }
        c_index++;
    }

    while(i < a_end){
        columns_C[c_index] = columns_A[i];
        values_C[c_index]  = values_A[i];;
        i++;
        c_index++;
    }
    while(j < b_end){
        columns_C[c_index] = columns_B[j];
        values_C[c_index]  = values_B[j];
        j++;
        c_index++;
    }
}