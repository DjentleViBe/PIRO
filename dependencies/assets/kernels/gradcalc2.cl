__kernel void gradcalc2(__global float *B,
                        __global float *C,
                        __global float *ind,
                        float delta_x,
                        float delta_y,
                        float delta_z,
                        uint nx, uint ny,
                        float deltat,
                        uint size) {
    uint id = get_global_id(0);
    if (id < size) {
        uint z = id / (nx * ny);
        uint y = (id / nx) % ny;
        uint x = id % nx;
        if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < (size / (nx * ny)) - 1) {
            C[id] = deltat * (ind[id] * ((B[id + 1] - B[id - 1]) / (2 * delta_x))
                    + ind[size + id] * ((B[id + nx] - B[id - nx]) / (2 * delta_y))
                    + ind[size * 2 + id] * ((B[id + nx * ny] - B[id - nx * ny]) / (2 * delta_z)));
        
            }
        }
}