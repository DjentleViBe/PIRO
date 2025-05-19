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

__kernel void SpMSCSR(const int M,
                        const int N,
                        __global float* values_A,
                        __global const int* columns_A,
                        __global const int* rowPointers_A,
                        __global float* B,
                        const int ind) {
    int index = get_global_id(0);
    if(index > M) return;
    int row = N * ind + findRow(index, rowPointers_A, M);
   
    if (row >= 0) {
        values_A[index] *= B[row];
    }
                
}