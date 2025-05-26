__kernel void residuals(__global float* input, 
                        __global float* output, 
                        __local float* scratch,
                        const int N){
    
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    if(gid > N) return;
    scratch[lid] = 0.0f;
    // Load input into local memory
    float val = input[gid];
    // if (isnan(val) || isinf(val)) val = 0.0f;
    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction in local memory
    for (int offset = group_size / 2; offset > 0; offset >>= 1) {
        if (lid < offset) {
            scratch[lid] += scratch[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result of this work-group to output
    if (lid == 0) {
        output[get_group_id(0)] = scratch[0];
    }
}