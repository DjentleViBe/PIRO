__kernel void gradcalc4(__global float *B,
                        __global float *C,
                        __global float *D,
                        float delta_x,
                        float delta_y,
                        float delta_z,
                        uint nx, uint ny,
                        float deltat,
                        uint size) {
    uint id = get_global_id(0);
    uint z = id / (nx * ny);
    uint y = (id / nx) % ny;
    uint x = id % nx;
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < (size / (nx * ny)) - 1) {
        if (id < size) {
            C[id] = deltat *    ((D[id + 1] - D[id - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_x) + 
                                (D[size + id + 1] - D[size + id - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_y) + 
                                (D[size * 2 + id + 1] - D[size * 2 + id - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_z));
        }
        else if(id > size && id < 2 * size){
            C[id] = deltat *    ((D[id - size + 1] - D[id - size - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_x) + 
                                (D[id + 1] - D[id - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_y) + 
                                (D[size + id + 1] - D[size + id - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_z));
        
        }
        else{
            C[id] = deltat *    ((D[id - 2 * size + 1] - D[id - 2 * size - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_x) + 
                                (D[id - size + 1] - D[id - size - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_y) + 
                                (D[id + 1] - D[id - 1]) * (B[id + 1] - B[id - 1]) / (2 * delta_z));
        
        }
    }
}