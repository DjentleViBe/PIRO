#include <kernelcsrgeam.hpp>
#include <vector>
#include <datatypes.hpp>
#include <preprocess.hpp>
#include <openclutilities.hpp>
#include <operatoroverload.hpp>

void Piro::kernelmethods::csrgeam(std::vector<CLBuffer> partA, std::vector<CLBuffer> partB){
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    // part A is vector field, part B is a matrix. Both size 3 * N
    if(Piro::INIT::getInstance().DOT_INIT == false){
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        Piro::kernels& kernels = Piro::kernels::getInstance();
        auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) +  1];
        int nnz = cd.values.size();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
        int N = n[0] * n[1] * n[2];
        
        Piro::CLBuffer partF;
        std::vector<Piro::CLBuffer> partE(3);
        cl_int err;
        partF.buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_WRITE,
                        sizeof(int) * (N), NULL, &err);
        partE[2].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_WRITE,
                        sizeof(float) * nnz * 3, NULL, &err);
        partE[1].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_WRITE,
                        sizeof(int) * nnz * 3, NULL, &err);
        partE[0].buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT), CL_MEM_READ_WRITE,
                        sizeof(int) * (N + 1), NULL, &err);
        
        // all 3 sparse matrices need to be added
        float patternf = 0.0f;
        int patterni = 0;
        cl_event event[3];
        clEnqueueFillBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), partE[2].buffer, &patternf, sizeof(float), 0, sizeof(float)*nnz*3, 0, NULL, &event[0]);
        clEnqueueFillBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), partE[1].buffer, &patterni, sizeof(int), 0, sizeof(int)*nnz*3, 0, NULL, &event[1]);
        clEnqueueFillBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), partE[0].buffer, &patterni, sizeof(int), 0, sizeof(int)*(N + 1), 0, NULL, &event[2]);
        clWaitForEvents(3, event);
        
        size_t globalWork[1] = { (size_t) N};  
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[19], 0, sizeof(cl_mem), &partB[1].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[19], 1, sizeof(cl_mem), &partB[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[19], 2, sizeof(cl_mem), &partB[4].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[19], 3, sizeof(cl_mem), &partB[3].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[19], 4, sizeof(cl_mem), &partB[7].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[19], 5, sizeof(cl_mem), &partB[6].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[19], 6, sizeof(cl_mem), &partF.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[19], 7, sizeof(cl_int), &N);

        clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[19], 1, NULL, globalWork, NULL, 0, NULL, NULL);
        clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
        clEnqueueCopyBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), partF.buffer, partE[0].buffer, 0, sizeof(int), sizeof(int)*(N), 0, NULL, NULL);
        for(int row = 0; row < N; row++){
            clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[21], 0, sizeof(cl_mem), &partF.buffer);
            clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[21], 1, sizeof(cl_mem), &partE[0].buffer);
            clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[21], 2, sizeof(cl_int), &N);
            clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[21], 3, sizeof(cl_int), &row);
            clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[21], 1, NULL, globalWork, NULL, 0, NULL, NULL);
            clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
        }

        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 0, sizeof(cl_mem), &partB[2].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 1, sizeof(cl_mem), &partB[1].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 2, sizeof(cl_mem), &partB[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 3, sizeof(cl_mem), &partB[5].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 4, sizeof(cl_mem), &partB[4].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 5, sizeof(cl_mem), &partB[3].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 6, sizeof(cl_mem), &partB[8].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 7, sizeof(cl_mem), &partB[7].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 8, sizeof(cl_mem), &partB[6].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 9, sizeof(cl_mem), &partE[2].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 10, sizeof(cl_mem), &partE[1].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 11, sizeof(cl_mem), &partE[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 12, sizeof(cl_mem), &partF.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 13, sizeof(cl_mem), &partA[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 14, sizeof(cl_int), &N);

        clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[22], 1, NULL, globalWork, NULL, 0, NULL, NULL);
        clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
        
        CDGPU.setvalue(Piro::CellDataGPU::RHS, partE);
        Piro::INIT::getInstance().DOT_INIT = true;
    }
}