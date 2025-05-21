#include <operatorlaplacian.hpp>
#include <CL/opencl.h>
#include <operatoroverload.hpp>
#include <datatypes.hpp>
#include <mathoperations.hpp>
#include <openclutilities.hpp>
#include <preprocess.hpp>
#include <bc.hpp>
#include <logger.hpp>
#include <methods.hpp>

int Piro::CSR::laplacian_cd(){
    Piro::CLBuffer CD_GPU;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    int N = n[0] * n[1] * n[2];
    Piro::CellData CD;
    std::vector<CLBuffer> laplacian_collect;
    // MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
    laplacian_collect.push_back(CD_GPU);
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].type = 2; // CSR
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers.assign(N + 1, 0.0);
    laplacian_collect.push_back(CD_GPU);
    laplacian_collect.push_back(CD_GPU);
    float norm = pow((l[0] / float(n[0] - 2)), 2);
    // Iterate over all grid points
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(i);
                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x-1, y, z, n[0], n[1]));
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x+1, y, z, n[0], n[1]));
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y-1, z, n[0], n[1]));
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    
                }
                if (y < n[1] - 1) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y+1, z, n[0], n[1]));
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    
                }

                // Neighbors in the z-direction (z±1)
                if (z > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y, z-1, n[0], n[1]));
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    
                }
                if (z < n[2] - 1) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y, z+1, n[0], n[1]));
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                }
            
                if (Piro::bc::isBoundaryPoint(x, y, z)) {
                    auto it = std::find(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.begin(),
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.end(), i);
    
                    if (it == MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.end()) {
                        // Add the boundary point if not already in the matrix.
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(i); // for diagonal entry
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(10.0); // large value to enforce the BC
                    }
                    else{
                        size_t index = std::distance(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.begin(), it);
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values[index] = 10.0;
                    }
                }

                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers[i + 1] = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.size();
            }
        }
    }
    
    cl_int err;
    laplacian_collect[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers.size(), 
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers.data(), &err);

    laplacian_collect[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) *  MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.size(), 
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.data(), &err);

    laplacian_collect[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) *  MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.size(), 
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.data(), &err);
    
    Piro::INIT::getInstance().LAP_INIT = true;

    CDGPU.setvalue(Piro::CellDataGPU::LAPLACIAN_CSR, laplacian_collect);
    Piro::Equation::getInstance().sparsecount += MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.size();
    return 0;
}

int Piro::DENSE::laplacian_cd(){
    Piro::CellData CD;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    std::vector<float> delta = SP.getvalue<std::vector<float>>(Piro::SolveParams::DELTA);
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    size_t total_size = n[0] * n[1] * n[2];
    float norm = pow((l[0] / float(n[0] - 2)), 2);
    // MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].type = 4; // DENSE
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].matrix.resize(total_size * total_size);

    for (int k = 0; k < n[2]; ++k) {
        for (int j = 0; j < n[1]; ++j) {
            for (int i = 0; i < n[0]; ++i) {
                int l = Piro::math_operations::idx(i, j, k, n[0], n[1]);

                // Diagonal entry
                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) 
                    + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].matrix[l * total_size + l] = -6 / norm;
                //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + l] = 0;

                // Off-diagonal entries
                if (i > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) 
                        + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].matrix[l * total_size + Piro::math_operations::idx(i-1, j, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l* total_size + Piro::math_operations::idx(i-1, j, k, n[0], n[1])] = -timestep / (2 * delta[0]);
                    
                }
                if (i < n[0] - 1) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) 
                        + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].matrix[l * total_size + Piro::math_operations::idx(i+1, j, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + Piro::math_operations::idx(i+1, j, k, n[0], n[1])] = timestep / (2 * delta[0]);
                    
                }
                if (j > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) 
                        + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].matrix[l * total_size + Piro::math_operations::idx(i, j-1, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + Piro::math_operations::idx(i, j-1, k, n[0], n[1])] = -timestep / (2 * delta[1]);
                    
                }
                if (j < n[1] - 1) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) 
                        + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].matrix[l * total_size + Piro::math_operations::idx(i, j+1, k, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l * total_size + Piro::math_operations::idx(i, j+1, k, n[0], n[1])] = timestep / (2 * delta[1]);
                    
                }
                if (k > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) 
                        + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].matrix[l* total_size + Piro::math_operations::idx(i, j, k-1, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l* total_size + Piro::math_operations::idx(i, j, k-1, n[0], n[1])] = -timestep / (2 * delta[2]);
                    
                }
                if (k < n[2] - 1) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) 
                        + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].matrix[l* total_size + Piro::math_operations::idx(i, j, k+1, n[0], n[1])] = 1 / norm;
                    //MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + 1].matrix[l* total_size + Piro::math_operations::idx(i, j, k+1, n[0], n[1])] = timestep / (2 * delta[2]);
                    
                }
            }
        }
    }
    // put the matrix into CL buffer
    cl_int err;
    Piro::CLBuffer laplacian;
    Piro::kernels& kernels = Piro::kernels::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    laplacian.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].matrix.size(), 
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].matrix.data(),
        &err);

    Piro::INIT::getInstance().LAP_INIT = true;
    CDGPU.setvalue(Piro::CellDataGPU::LAPLACIAN_DENSE, laplacian);
    return 0;
}

int Piro::COO::laplacian_cd(){
    Piro::CLBuffer CD_GPU;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    int N = n[0] * n[1] * n[2];
    Piro::CellData CD;
    std::vector<CLBuffer> laplacian_collect;
    // MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
    laplacian_collect.push_back(CD_GPU);
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].type = 3; // COO
    // MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers.assign(N + 1, 0.0);
    laplacian_collect.push_back(CD_GPU);
    laplacian_collect.push_back(CD_GPU);
    float norm = pow((l[0] / float(n[0] - 2)), 2);
    // Iterate over all grid points
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rows.push_back(i);
                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(i % N);
                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rows.push_back(i);
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x-1, y, z, n[0], n[1]) % N);
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rows.push_back(i);
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x+1, y, z, n[0], n[1]) % N);
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rows.push_back(i);
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y-1, z, n[0], n[1]) % N);
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    
                }
                if (y < n[1] - 1) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rows.push_back(i);
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y+1, z, n[0], n[1]) % N);
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    
                }

                // Neighbors in the z-direction (z±1)
                if (z > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rows.push_back(i);
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y, z-1, n[0], n[1]) % N);
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    
                }
                if (z < n[2] - 1) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rows.push_back(i);
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y, z+1, n[0], n[1]) % N);
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                }
                
                if (Piro::bc::isBoundaryPoint(x, y, z)) {
                    auto it = std::find(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.begin(),
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.end(), i);
    
                    if (it == MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.end()) {
                        // Add the boundary point if not already in the matrix.
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rows.push_back(i); // for diagonal entry
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(i % N); // for diagonal entry
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(10.0); // large value to enforce the BC
                    }
                    else{
                        size_t index = std::distance(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.begin(), it);
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values[index] = 10.0;
                    }
                }

                // MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers[i + 1] = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.size();
            }
        }
    }
    
    cl_int err;
    laplacian_collect[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rows.size(), 
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rows.data(), &err);

    laplacian_collect[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) *  MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.size(), 
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.data(), &err);

    laplacian_collect[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) *  MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.size(), 
        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.data(), &err);
    
    Piro::INIT::getInstance().LAP_INIT = true;

    CDGPU.setvalue(Piro::CellDataGPU::LAPLACIAN_COO, laplacian_collect);
    return 0;
}

int Piro::HT::laplacian_cd(){
    Piro::CLBuffer CD_GPU;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    int N = n[0] * n[1] * n[2];
    Piro::CellData CD;
    // MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD.push_back(CD);
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].type = 2; // CSR
    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers.assign(N + 1, 0.0);
    float norm = pow((l[0] / float(n[0] - 2)), 2);
    // Iterate over all grid points
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                
                // Self connection (central point)
                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(i);
                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(-6.0/norm);
                
                // Neighbors in the x-direction (x±1)
                if (x > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x-1, y, z, n[0], n[1]));
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    
                }
                if (x < n[0] - 1) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x+1, y, z, n[0], n[1]));
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                
                }

                // Neighbors in the y-direction (y±1)
                if (y > 0) {
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y-1, z, n[0], n[1]));
                    MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    
                }
                    if (y < n[1] - 1) {
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y+1, z, n[0], n[1]));
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                        
                    }

                    // Neighbors in the z-direction (z±1)
                    if (z > 0) {
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y, z-1, n[0], n[1]));
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                        
                    }
                    if (z < n[2] - 1) {
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(math_operations::index(x, y, z+1, n[0], n[1]));
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(1.0/norm);
                    }
                
                    if (Piro::bc::isBoundaryPoint(x, y, z)) {
                        auto it = std::find(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.begin(),
                        MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.end(), i);
        
                        if (it == MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.end()) {
                            // Add the boundary point if not already in the matrix.
                            MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.push_back(i); // for diagonal entry
                            MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.push_back(10.0); // large value to enforce the BC
                        }
                        else{
                            size_t index = std::distance(MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.begin(), it);
                            MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values[index] = 10.0;
                        }
                    }

                MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].rowpointers[i + 1] = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].columns.size();
            }
        }
    }
    Piro::Equation::getInstance().sparsecount += MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)].values.size();
    Piro::logger::debug("Generating hash table");
    
    float load = SP.getvalue<float>(Piro::SolveParams::A) * pow(N, SP.getvalue<float>(Piro::SolveParams::B)) + SP.getvalue<float>(Piro::SolveParams::C);
    SP.setvalue(Piro::SolveParams::TABLE_SIZE, (int)(Piro::Equation::getInstance().sparsecount / (load * SP.getvalue<float>(Piro::SolveParams::LOADFACTOR))));
    auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM)];
    
    Piro::logger::debug("Hash Table size : ", SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE));
    cd.Hash_keys_V.assign(SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE), -1);
    cd.Hash_val_V.assign(SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE), 0.0f);
    
    
    int ind;
    for (int i = 0; i < N; i++){
        int start = cd.rowpointers[i];
        int end = cd.rowpointers[i + 1];
        for(int j = start; j < end; j++){
            int col = cd.columns[j];
            int row = i;

            ind = row * N + col;
            Piro::methods::sethash(ind, cd.values[j], SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE), cd.Hash_keys_V, cd.Hash_val_V);
        }
    }
    cl_int err;
    cl_event event0, event1;
    std::vector<CLBuffer> laplacian_ht;
    laplacian_ht.push_back(CD_GPU);
    laplacian_ht.push_back(CD_GPU);

    laplacian_ht[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_WRITE , sizeof(int) * SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE), nullptr, &err);
    laplacian_ht[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_WRITE , sizeof(float) * SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE), nullptr, &err);
    
    
    CDGPU.setvalue(Piro::CellDataGPU::LAPLACIAN_HT, laplacian_ht);
    // hash_to_dense_and_print(Hash_keys_V, Hash_val_V, N, TABLE_SIZE);
    err = clEnqueueWriteBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_HT)[0].buffer, CL_FALSE, 
                                0, 
                                sizeof(int) * SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE),
                                cd.Hash_keys_V.data(), 
                                0, nullptr, &event0);
    // assert(Lap_rowptr_V.data() + rowouter != nullptr);
    
    err = clEnqueueWriteBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), CDGPU.getvalue<std::vector<Piro::CLBuffer>>(Piro::CellDataGPU::LAPLACIAN_HT)[1].buffer, CL_FALSE, 
                                0, 
                                sizeof(float) * SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE),
                                cd.Hash_val_V.data(), 
                                0, nullptr, &event1);

    // Wait for all transfers to complete
    clWaitForEvents(2, (cl_event[]){event0, event1});
    Piro::INIT::getInstance().LAP_INIT = true;
    return 0;
}