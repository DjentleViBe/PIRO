__kernel void filter_row(
        __global const float* zeroArray,
        __global float* outputArray,
        __global float* rowArray,
        const int n,
        __global float* pivot,
        const int row
    ) {
        int gid = get_global_id(0);
        if(gid < n){
            outputArray[gid] = (zeroArray[gid] * rowArray[row]) / pivot[0];
        }
    }