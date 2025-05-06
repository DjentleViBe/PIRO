__kernel void lu_decompose_sparse(__global float* A, 
                            __global float* L,
                            int N,
                            __global float* Avalues,
                            __global int* Aind,
                            __global int* Arowptr) {
        // Get the global row index
        unsigned int i = get_global_id(0);
        unsigned int j = get_global_id(1);
    
       if (i < N && j < N) {
            // Forward elimination
            for (int k = 0; k < N - 1; k++) {
                float pivot = A[k * N + k];                             // Pivot element at A[k, k]
                float factor = A[i * N + k] / pivot;                    // Calculate factor to eliminate
                if (factor != 0){
                    if (i > k && j < N) {                               // Only update rows below the diagonal
                        if (i > k) {                                    // Process rows below the diagonal
                            A[i * N + j] -= factor * A[k * N + j];      // Subtract row k from row i
                        }
                    }
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE);                              // Ensure all threads are synchronized for the next iteration
        }
    }