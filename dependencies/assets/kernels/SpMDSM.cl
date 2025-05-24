int findRow(const int index, __global const int* rowPointers_A, const int M) {
    int low = 0;
    int high = M;

    while (low < high) {
        int mid = (low + high) / 2;
        if (rowPointers_A[mid] <= index) {
            if (index < rowPointers_A[mid + 1])
                return mid;
            else
                low = mid + 1;
        }
        else {
            high = mid;
        }
    }
    return -1; // not found
}

__kernel void SpMDSM(__global float* values_A,
                        __global const int* columns_A,
                        __global const int* rowPointers_A,
                        __global const float* diag,
                        const int nnz, const int M, const int ind){

    int index = get_global_id(0);
    // if(index > nnz) return;
    int row = findRow(index, rowPointers_A, M);
    if(row == columns_A[index]){
        values_A[index] *= diag[M * ind + row];
    }
}
