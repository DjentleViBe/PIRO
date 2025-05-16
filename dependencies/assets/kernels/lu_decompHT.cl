void lookup(int index, float* val, int* hk, __global float* Hash_val_V, __global int* Hash_Keys_V, int TABLE_SIZE){
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

__kernel void lu_decompHT(
    __global int* hashkey,
    __global float* hashvalue,
    const int N,
    const int rowouter,

    const int TABLE_SIZE,
    const int limit
) {
    // Get the global ID for this work item
    const int gid = get_global_id(0);
    if(gid > limit){
        return;
    }

    // printf("%d ", gid);
    int current_row = (gid / (N - rowouter)) + rowouter + 1;
    int current_col = (gid % (N - rowouter)) + rowouter;
    int index_current = current_row * N + current_col;
    int index_0 = rowouter * N + current_col;
    int index_piv = rowouter * N + rowouter;
    int index_factor = current_row * N + rowouter;


    int hash_index_current, hash_index_0, hash_index_piv, hash_index_factor;
    float val, val_0, piv, factor;

    lookup(index_current, &val, &hash_index_current, hashvalue, hashkey, TABLE_SIZE);
    lookup(index_0, &val_0, &hash_index_0, hashvalue, hashkey, TABLE_SIZE);
    lookup(index_piv, &piv, &hash_index_piv, hashvalue, hashkey, TABLE_SIZE);
    lookup(index_factor, &factor, &hash_index_factor, hashvalue, hashkey, TABLE_SIZE);
    // printf("gid = %d , val = %f, val_0 = %f, index = %d, hash_index_current = %d, piv = %f, factor = %d, %f\n", gid, val, val_0, index_current, hash_index_current, piv, hash_index_factor, factor);

    if(piv != 0 && factor != 0){
        val = val - (factor / piv) * val_0;
        hashvalue[hash_index_current] = val;
        if(fabs(val) < 1E-6){
            hashkey[hash_index_current] = -2;
        }
    }
}