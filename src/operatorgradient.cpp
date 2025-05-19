#include <operatorgradient.hpp>
#include <CL/opencl.h>
#include <operatoroverload.hpp>
#include <datatypes.hpp>
#include <mathoperations.hpp>
#include <openclutilities.hpp>
#include <preprocess.hpp>
#include <bc.hpp>
#include <logger.hpp>
#include <methods.hpp>

int Piro::CSR::gradient_cd(){
    // lets do just X direction
    cl_int err;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    int N = n[0] * n[1] * n[2];
    Piro::CellData CD;
    // Piro::logger::info(N);
    float norm = 2 * (l[0] / float(n[0] - 2));
    auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) +  1];
    cd.rowpointers.assign(N + 1, 0);

    Piro::CLBuffer CD_GPU;
    std::vector<CLBuffer> gradient_collect_x;
    gradient_collect_x.push_back(CD_GPU);
    gradient_collect_x.push_back(CD_GPU);
    gradient_collect_x.push_back(CD_GPU);
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // Piro::logger::info(i);
                // index right side 
                if (x > 0) {
                    int right = math_operations::index(x - 1, y, z, n[0], n[1]); 
                    cd.columns.push_back(right);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (x < n[0] - 1) {
                    int left = math_operations::index(x + 1, y, z, n[0], n[1]);
                    cd.columns.push_back(left);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
    
    gradient_collect_x[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) * cd.rowpointers.size(), 
                                                    cd.rowpointers.data(), &err);

    gradient_collect_x[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) *  cd.columns.size(), 
                                                    cd.columns.data(), &err);

    gradient_collect_x[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(float) *  cd.values.size(), 
                                                    cd.values.data(), &err);
    cd.columns.clear();
    cd.values.clear();
    std::vector<CLBuffer> gradient_collect_y;
    gradient_collect_y.push_back(CD_GPU);
    gradient_collect_y.push_back(CD_GPU);
    gradient_collect_y.push_back(CD_GPU);
    // Piro::logger::info("Gradient gen started");
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // index right side 
                if (y > 0) {
                    int top = math_operations::index(x, y - 1, z, n[0], n[1]); 
                    cd.columns.push_back(top);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (y < n[1] - 1) {
                    int bottom = math_operations::index(x, y + 1, z, n[0], n[1]);
                    cd.columns.push_back(bottom);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
    gradient_collect_y[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) * cd.rowpointers.size(), 
                                                    cd.rowpointers.data(), &err);

    gradient_collect_y[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) *  cd.columns.size(), 
                                                    cd.columns.data(), &err);

    gradient_collect_y[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(float) *  cd.values.size(), 
                                                    cd.values.data(), &err);
    cd.columns.clear();
    cd.values.clear();
    std::vector<CLBuffer> gradient_collect_z;
    gradient_collect_z.push_back(CD_GPU);
    gradient_collect_z.push_back(CD_GPU);
    gradient_collect_z.push_back(CD_GPU);
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // Piro::logger::info(i);
                // index right side 
                if (z > 0) {
                    int up = math_operations::index(x, y, z - 1, n[0], n[1]); 
                    // Piro::logger::info(right);
                    cd.columns.push_back(up);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (z < n[2] - 1) {
                    int down = math_operations::index(x, y, z + 1, n[0], n[1]);
                    cd.columns.push_back(down);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
    gradient_collect_z[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) * cd.rowpointers.size(), 
                                                    cd.rowpointers.data(), &err);

    gradient_collect_z[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) *  cd.columns.size(), 
                                                    cd.columns.data(), &err);

    gradient_collect_z[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(float) *  cd.values.size(), 
                                                    cd.values.data(), &err);
    std::vector<Piro::CLBuffer> gradient_collect;
    gradient_collect.reserve(gradient_collect_x.size() + gradient_collect_y.size() + gradient_collect_z.size());
    gradient_collect.insert(gradient_collect.end(), gradient_collect_x.begin(), gradient_collect_x.end());
    gradient_collect.insert(gradient_collect.end(), gradient_collect_y.begin(), gradient_collect_y.end());
    gradient_collect.insert(gradient_collect.end(), gradient_collect_z.begin(), gradient_collect_z.end());
    CDGPU.setvalue(Piro::CellDataGPU::GRADIENT_CSR, gradient_collect);
    Piro::INIT::getInstance().GRAD_INIT = true;
    return 0;
}

int Piro::CSR::gradient_uw(){
    // lets do just X direction
    cl_int err;
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    std::vector<float> l = MP.getvalue<std::vector<float>>(Piro::MeshParams::L);
    int N = n[0] * n[1] * n[2];
    Piro::CellData CD;
    // Piro::logger::info(N);
    float norm = (l[0] / float(n[0] - 2));
    auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) +  1];
    cd.rowpointers.assign(N + 1, 0);

    Piro::CLBuffer CD_GPU;
    std::vector<CLBuffer> gradient_collect_x;
    gradient_collect_x.push_back(CD_GPU);
    gradient_collect_x.push_back(CD_GPU);
    gradient_collect_x.push_back(CD_GPU);
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // Piro::logger::info(i);
                // index right side 
                if (x > 0) {
                    int right = math_operations::index(x - 1, y, z, n[0], n[1]); 
                    cd.columns.push_back(right);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (x < n[0] - 1) {
                    int left = math_operations::index(x, y, z, n[0], n[1]);
                    cd.columns.push_back(left);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
    Piro::logger::info("X num", cd.values.size());
    gradient_collect_x[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) * cd.rowpointers.size(), 
                                                    cd.rowpointers.data(), &err);

    gradient_collect_x[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) *  cd.columns.size(), 
                                                    cd.columns.data(), &err);

    gradient_collect_x[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(float) *  cd.values.size(), 
                                                    cd.values.data(), &err);
    cd.columns.clear();
    cd.values.clear();
    std::vector<CLBuffer> gradient_collect_y;
    gradient_collect_y.push_back(CD_GPU);
    gradient_collect_y.push_back(CD_GPU);
    gradient_collect_y.push_back(CD_GPU);
    // Piro::logger::info("Gradient gen started");
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // index right side 
                if (y > 0) {
                    int top = math_operations::index(x, y - 1, z, n[0], n[1]); 
                    cd.columns.push_back(top);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (y < n[1] - 1) {
                    int bottom = math_operations::index(x, y, z, n[0], n[1]);
                    cd.columns.push_back(bottom);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
    Piro::logger::info("Y num", cd.values.size());
    gradient_collect_y[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) * cd.rowpointers.size(), 
                                                    cd.rowpointers.data(), &err);

    gradient_collect_y[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) *  cd.columns.size(), 
                                                    cd.columns.data(), &err);

    gradient_collect_y[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(float) *  cd.values.size(), 
                                                    cd.values.data(), &err);
    cd.columns.clear();
    cd.values.clear();
    std::vector<CLBuffer> gradient_collect_z;
    gradient_collect_z.push_back(CD_GPU);
    gradient_collect_z.push_back(CD_GPU);
    gradient_collect_z.push_back(CD_GPU);
    for (int z = 0; z < n[2]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            for (int x = 0; x < n[0]; ++x) {
                int i = math_operations::index(x, y, z, n[0], n[1]); // 1D index of the (x, y, z) point
                // Piro::logger::info(i);
                // index right side 
                if (z > 0) {
                    int up = math_operations::index(x, y, z - 1, n[0], n[1]); 
                    // Piro::logger::info(right);
                    cd.columns.push_back(up);
                    cd.values.push_back(-1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
                }
                // index left side
                if (z < n[2] - 1) {
                    int down = math_operations::index(x, y, z, n[0], n[1]);
                    cd.columns.push_back(down);
                    cd.values.push_back(1/norm);
                    cd.rowpointers[i + 1] = cd.columns.size();
            
                }
            }
        }
    }
    Piro::logger::info("Z num", cd.values.size());
    gradient_collect_z[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) * cd.rowpointers.size(), 
                                                    cd.rowpointers.data(), &err);

    gradient_collect_z[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(int) *  cd.columns.size(), 
                                                    cd.columns.data(), &err);

    gradient_collect_z[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                    sizeof(float) *  cd.values.size(), 
                                                    cd.values.data(), &err);
    std::vector<Piro::CLBuffer> gradient_collect;
    gradient_collect.reserve(gradient_collect_x.size() + gradient_collect_y.size() + gradient_collect_z.size());
    gradient_collect.insert(gradient_collect.end(), gradient_collect_x.begin(), gradient_collect_x.end());
    gradient_collect.insert(gradient_collect.end(), gradient_collect_y.begin(), gradient_collect_y.end());
    gradient_collect.insert(gradient_collect.end(), gradient_collect_z.begin(), gradient_collect_z.end());
    CDGPU.setvalue(Piro::CellDataGPU::GRADIENT_CSR, gradient_collect);
    Piro::INIT::getInstance().GRAD_INIT = true;
    return 0;
}