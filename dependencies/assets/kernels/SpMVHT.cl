inline void lookup(int index, float* val, int* hk, __global float* Hash_val_V, __global int* Hash_Keys_V, const int TABLE_SIZE){
        int hash_index = index % TABLE_SIZE;
        int attempts = 0;
        while (Hash_Keys_V[hash_index] != -1) {
            if (Hash_Keys_V[hash_index] == index) {
                *val = Hash_val_V[hash_index]; // key found
                *hk = hash_index;
                return;
            }
            hash_index = (hash_index + 1) % TABLE_SIZE;
            attempts++;
            if(attempts >= TABLE_SIZE){
                break;
            }
        }

    // Key not found
    *val = 0.0f;
    *hk = -1;
    }

__kernel void SpMVHT(__global int *hashkey,
                        __global float *hashvalue,
                        __global const float *B,
                        __global float *C, int N, const int TABLE_SIZE, float deltat){

    int row = get_global_id(0);  // Each work item handles one row
    float val;
    int hash_index;
    if(row < N){
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            int index = row * N + col;

            lookup(index, &val, &hash_index, hashvalue, hashkey, TABLE_SIZE);
            sum += val * B[col];
        }
        C[row] = deltat * sum;
    }
}