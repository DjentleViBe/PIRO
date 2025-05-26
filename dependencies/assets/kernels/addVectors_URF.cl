__kernel void addVectors_URF(__global float *A,
                        __global float *C,
                        const float URF,
                        uint size) {
        uint id = get_global_id(0);
        if(id > size) return;
        float temp = A[id];
        A[id] += C[id] * URF;
        C[id] = A[id] - temp;
        
    }