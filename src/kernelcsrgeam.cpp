#include <kernelcsrgeam.hpp>
#include <vector>
#include <datatypes.hpp>
#include <preprocess.hpp>
#include <openclutilities.hpp>
#include <operatoroverload.hpp>
#include <printutilities.hpp>

void Piro::kernelmethods::csrgeam(std::vector<CLBuffer> partA, std::vector<CLBuffer> partB, int module){
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    // part A is vector field, part B is a matrix. Both size 3 * N
    if(Piro::INIT::getInstance().DOT_INIT == false){
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        Piro::kernels& kernels = Piro::kernels::getInstance();
        // choose which sparse matrix
        auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + MP.getvalue<int>(Piro::MeshParams::CONSTANTNUM) + module];
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

void Piro::kernelmethods::csrgeam_2(std::vector<CLBuffer> partA, std::vector<CLBuffer> partB, int module){
    Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
    // part A is vector field, part B is a matrix. Both size 3 * N
    if(Piro::INIT::getInstance().DOT_INIT == false){
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        Piro::kernels& kernels = Piro::kernels::getInstance();
        // choose which sparse matrix
        auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + MP.getvalue<int>(Piro::MeshParams::CONSTANTNUM) + module];
        int nnz = cd.values.size();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
        int N = n[0] * n[1] * n[2];
        
        // scale partB by diag(Ux, Uy, Uz)
        size_t globalWorkSize[1] = { (size_t) nnz}; 
        int comp = 0;
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 0, sizeof(cl_mem), &nnz);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 1, sizeof(cl_mem), &N);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 2, sizeof(cl_mem), &partB[2].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 3, sizeof(cl_mem), &partB[1].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 4, sizeof(cl_mem), &partB[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 5, sizeof(cl_mem), &partA[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 6, sizeof(cl_mem), &comp);
        clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        
        comp = 1;
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 0, sizeof(cl_mem), &nnz);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 1, sizeof(cl_mem), &N);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 2, sizeof(cl_mem), &partB[5].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 3, sizeof(cl_mem), &partB[4].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 4, sizeof(cl_mem), &partB[3].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 5, sizeof(cl_mem), &partA[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 6, sizeof(cl_mem), &comp);
        clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        
        comp = 2;
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 0, sizeof(cl_mem), &nnz);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 1, sizeof(cl_mem), &N);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 2, sizeof(cl_mem), &partB[8].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 3, sizeof(cl_mem), &partB[7].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 4, sizeof(cl_mem), &partB[6].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 5, sizeof(cl_mem), &partA[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 6, sizeof(cl_mem), &comp);
        clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
        
        /*
        std::vector<int> rp = Piro::opencl_utilities::copyCL<int>(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE),
                                        partB[0].buffer, N , NULL);
        std::vector<int> col = Piro::opencl_utilities::copyCL<int>(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE),
                                        partB[1].buffer, nnz , NULL); 
        std::vector<float> val = Piro::opencl_utilities::copyCL<float>(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE),
                                        partB[2].buffer, nnz , NULL);         
        Piro::print_utilities::csr_to_dense_and_print(rp, col, val, N);*/

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

        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 0, sizeof(cl_mem), &partB[2].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 1, sizeof(cl_mem), &partB[1].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 2, sizeof(cl_mem), &partB[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 3, sizeof(cl_mem), &partB[5].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 4, sizeof(cl_mem), &partB[4].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 5, sizeof(cl_mem), &partB[3].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 6, sizeof(cl_mem), &partB[8].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 7, sizeof(cl_mem), &partB[7].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 8, sizeof(cl_mem), &partB[6].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 9, sizeof(cl_mem), &partE[2].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 10, sizeof(cl_mem), &partE[1].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 11, sizeof(cl_mem), &partE[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 12, sizeof(cl_mem), &partF.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 13, sizeof(cl_int), &N);

        clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[20], 1, NULL, globalWork, NULL, 0, NULL, NULL);
        clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
        
        CDGPU.setvalue(Piro::CellDataGPU::RHS, partE);
        Piro::INIT::getInstance().DOT_INIT = true;
    }
}

void Piro::kernelmethods::csrscale(std::vector<CLBuffer> partA, std::vector<CLBuffer> partB, int module){

    if(Piro::INIT::getInstance().DOT_INIT == false){
        Piro::CellDataGPU& CDGPU = Piro::CellDataGPU::getInstance();
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        Piro::kernels& kernels = Piro::kernels::getInstance();
        // choose which sparse matrix
        auto& cd = MP.getvalue<std::vector<AMR>>(Piro::MeshParams::AMR)[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + MP.getvalue<int>(Piro::MeshParams::CONSTANTNUM) + module];
        int nnz = cd.values.size();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
        int N = n[0] * n[1] * n[2];
        
        // scale partB by diag(Ux, Uy, Uz)
        size_t globalWorkSize[1] = { (size_t) nnz}; 

        int comp = 0;
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[23], 0, sizeof(cl_mem), &nnz);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[23], 1, sizeof(cl_mem), &N);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[23], 2, sizeof(cl_mem), &partB[2].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[23], 3, sizeof(cl_mem), &partB[1].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[23], 4, sizeof(cl_mem), &partB[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[23], 5, sizeof(cl_mem), &partA[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[23], 6, sizeof(cl_mem), &comp);
        clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[17], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        Piro::INIT::getInstance().DOT_INIT = true;
        CDGPU.setvalue(Piro::CellDataGPU::RHS, partB);
    }
}