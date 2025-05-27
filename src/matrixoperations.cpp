#include <CL/opencl.h>
#include <logger.hpp>
#include <openclutilities.hpp>
#include <preprocess.hpp>
#include <methods.hpp>
#include <operatoroverload.hpp>
#include <cmath>
#include <matrixoperations.hpp>

void Piro::matrix_operations::HT::lu_decomposition(const std::vector<CLBuffer>& other){
    Piro::MeshParams& MP = Piro::MeshParams::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    Piro::kernels& kernels = Piro::kernels::getInstance();
    std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
    
    int N = n[0] * n[1] * n[2];
    int TS = SP.getvalue<int>(Piro::SolveParams::TABLE_SIZE);
    // int index = MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + MP.getvalue<int>(Piro::MeshParams::CONSTANTNUM);
    cl_event event0 = nullptr;
    cl_event event1 = nullptr; 
    cl_event event4, event5;
    cl_event events[] = {event0, event1};
    Piro::logger::debug("Buffer creation begin");
    /////////////////////////////////////////////////////////////////////////////////////////

    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[10], 0, sizeof(cl_mem), &other[0]);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[10], 1, sizeof(cl_mem), &other[1]);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[10], 2, sizeof(cl_int), &N);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[10], 4, sizeof(cl_int), &TS);
    Piro::logger::debug("Buffer creation end");
    size_t globalWorkSize[1];
    size_t localWorkSize[1];
    Piro::logger::debug("Loop begin");
    auto& cd = MP.getvalue<std::vector<AMR_LEVELS>>(Piro::MeshParams::AMRLEVELS)[0].amr[0].CD[MP.getvalue<int>(Piro::MeshParams::VECTORNUM) + MP.getvalue<int>(Piro::MeshParams::SCALARNUM) + MP.getvalue<int>(Piro::MeshParams::CONSTANTNUM)];

    int limit;
    for (int rowouter = 0; rowouter < N - 1; rowouter++){
        Piro::logger::info("rowouter :", rowouter, ", HashTable size :", TS);
        std::vector<int> rowouter_cols;

        for(int co = rowouter; co < N; co++){
            float valofrow = Piro::methods::lookup(rowouter, co, N, cd.Hash_keys_V, cd.Hash_val_V, TS);
            if(std::abs(valofrow) > 1E-6){
                rowouter_cols.push_back(co);
            }
        }
        Piro::logger::warning("row outer", rowouter_cols);
        
        for (int r = rowouter + 1; r < N; ++r) {
            if(Piro::methods::lookup(r, rowouter, N, cd.Hash_keys_V, cd.Hash_val_V, TS) == 0.0){
                continue;
            }

            for (int col : rowouter_cols) {
                // std::cout << col << ", ";
                // if(col < rowouter) continue;
                Piro::methods::lookupandset(r, col, N, 0.0f, cd.Hash_keys_V, cd.Hash_val_V, TS);
            }

        }
        
        Piro::logger::debug("Inserting 0s finished");
        clEnqueueWriteBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), 
                                    other[0].buffer, CL_FALSE, 
                                    0, 
                                    sizeof(int) * TS,
                                    cd.Hash_keys_V.data(), 
                                    0, nullptr, &event0);
        // assert(Lap_rowptr_V.data() + rowouter != nullptr);

        clEnqueueWriteBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), 
                                    other[1].buffer, CL_FALSE, 
                                    0, 
                                    sizeof(float) * TS,
                                    cd.Hash_val_V.data(), 
                                    0, nullptr, &event1);

        // Wait for all transfers to complete
        clWaitForEvents(2, events);
        Piro::logger::debug("Map memory object finished");
        // std::cout << cd.rowpointers[N] << std::endl;
        limit = (N - rowouter - 1) * (N - rowouter);
        // int mws = static_cast<int>(maxWorkGroupSize);
        size_t nnz = (size_t)limit;
        size_t local = (size_t)kernels.getvalue<cl_uint>(Piro::kernels::MAXWORKGROUPSIZE); // or whatever max workgroup size your device supports
        // Piro::logger::debug("nnz :", nnz, "local :", mws);
        globalWorkSize[0] = ((nnz + local - 1) / local) * local;
        localWorkSize[0] = local;
        // size_t localWorkSize[1] = { globalWorkSize[0] / 4 };
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[10], 5, sizeof(cl_int), &limit);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[10], 3, sizeof(cl_int), &rowouter);
        clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), 
                                    kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[10], 1, NULL, 
                                    globalWorkSize, localWorkSize, 0, NULL, NULL);
        clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
        Piro::logger::debug("Kernel finished");
        Piro::opencl_utilities::copyCL_offset<float>(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), other[1].buffer, cd.Hash_val_V, 0, TS, &event4);
        Piro::opencl_utilities::copyCL_offset<int>(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), other[0].buffer, cd.Hash_keys_V, 0, TS, &event5);
        Piro::logger::debug("CopyCL");
        Piro::logger::debug("Erased");
        
    }
    Piro::print_utilities::hash_to_dense_and_print(cd.Hash_keys_V, cd.Hash_val_V, N, TS);
}

