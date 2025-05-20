__kernel void csrgeam_1(__global const int* columns_A,
                        __global const int* rowPointers_A,
                        __global const int* columns_B,
                        __global const int* rowPointers_B,
                        __global const int* columns_C,
                        __global const int* rowPointers_C,
                        
                        __global int* phase_one,
                        const int num_rows){
    
    int row = get_global_id(0);
    if (row >= num_rows) return;

    int a_start = rowPointers_A[row];
    int a_end = rowPointers_A[row + 1];
    
    int b_start = rowPointers_B[row];
    int b_end = rowPointers_B[row + 1];

    int c_start = rowPointers_C[row];
    int c_end = rowPointers_C[row + 1];

    int i = a_start;
    int j = b_start;
    int k = c_start;
    int count = 0;
    while(i < a_end && j < b_end){
        
        int colA = columns_A[i];
        int colB = columns_B[j];
        int colC = columns_C[k];
        int min_col = min(colA, colB);

        if(colA == min_col){
            i++;
            count++;
        }
        else if(colB == min_col){
            j++;
            count++;
        }
        else{
            count++;
            i++;
            j++;
        }
    }

    while(i < a_end){
        count++;
        i++;
    }
    while(j < b_end){
        count++;
        j++;
    }

    phase_one[row] = count;
}