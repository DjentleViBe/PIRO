#include <CL/opencl.h>
#include <logger.hpp>
#include <openclutilities.hpp>
#include <preprocess.hpp>
#include <methods.hpp>
#include <operatoroverload.hpp>
#include <cmath>

namespace Piro::matrix_operations{

    void lu_decomposition_HTLF(const std::vector<CLBuffer>& other){
        int N = MP.n[0] * MP.n[1] * MP.n[2];
        int index = MP.vectornum + MP.scalarnum;
        cl_int err;
        cl_event event0 = nullptr;
        cl_event event1 = nullptr; 
        cl_event event4, event5, event7, event8, event9;
        cl_event events[] = {event0, event1};
        Piro::logger::debug("Buffer creation begin");
        
        MP.AMR[0].CD[index].rowpointers = Piro::opencl_utilities::copyCL<int>(queue, other[0].buffer, N + 1, &event7);
        MP.AMR[0].CD[index].columns = Piro::opencl_utilities::copyCL<int>(queue, other[1].buffer, RHS.sparsecount, &event8);
        MP.AMR[0].CD[index].values = Piro::opencl_utilities::copyCL<float>(queue, other[2].buffer, RHS.sparsecount, &event9);

        
        /////////////////////////////////////////////////////////////////////////////////////////
        auto& cd = MP.AMR[0].CD[index];
        CLBuffer LFvalues, LFkeys;
        float load = SP.a * pow(N, SP.b) + SP.c;
        // int TABLE_SIZE = nextPowerOf2(cd.columns.size() / load);
        // int raw_size = cd.columns.size() / load;
        // int TABLE_SIZE = next_prime(raw_size);
        int TABLE_SIZE = RHS.sparsecount / (load * SP.loadfactor);
        Piro::logger::debug("RHS_INIT begin, sparse count :", RHS.sparsecount, ", Table size :" , TABLE_SIZE, ", Load factor :", load * SP.loadfactor);
        Piro::logger::warning("N * N :", N*N);
        
        LFvalues.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(float) * TABLE_SIZE, nullptr, &err);
        LFkeys.buffer = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(int) * TABLE_SIZE, nullptr, &err);
        err |= clSetKernelArg(kernelfilterarray, 0, sizeof(cl_mem), &LFkeys.buffer);
        err |= clSetKernelArg(kernelfilterarray, 1, sizeof(cl_mem), &LFvalues.buffer);
        err |= clSetKernelArg(kernelfilterarray, 2, sizeof(cl_int), &N);
        err |= clSetKernelArg(kernelfilterarray, 4, sizeof(cl_int), &TABLE_SIZE);
        Piro::logger::debug("Buffer creation end");
        size_t globalWorkSize[1];
        size_t localWorkSize[1];
        int ind;
        Piro::logger::debug("Loop begin");
        
        std::vector<float> Hash_val_V(TABLE_SIZE, 0.0);
        std::vector<int> Hash_keys_V(TABLE_SIZE, -1);
        /////////////////////////////////////// Generate hash tables ///////////////////////////////////////////
        for (int i = 0; i < N; i++){
            int start = cd.rowpointers[i];
            int end = cd.rowpointers[i + 1];
            for(int j = start; j < end; j++){
                int col = cd.columns[j];
                int row = i;

                ind = row * N + col;
                Piro::methods::sethash(ind, cd.values[j], TABLE_SIZE, Hash_keys_V, Hash_val_V);
            }
        }
        // hash_to_dense_and_print(Hash_keys_V, Hash_val_V, N, TABLE_SIZE);
        err = clEnqueueWriteBuffer(queue, LFkeys.buffer, CL_FALSE, 
                                    0, 
                                    sizeof(int) * TABLE_SIZE,
                                    Hash_keys_V.data(), 
                                    0, nullptr, &event0);
        // assert(Lap_rowptr_V.data() + rowouter != nullptr);
        
        err = clEnqueueWriteBuffer(queue, LFvalues.buffer, CL_FALSE, 
                                    0, 
                                    sizeof(float) * TABLE_SIZE,
                                    Hash_val_V.data(), 
                                    0, nullptr, &event1);

        // Wait for all transfers to complete
        clWaitForEvents(2, events);
        int limit;
        for (int rowouter = 0; rowouter < N - 1; rowouter++){
            Piro::logger::info("rowouter :", rowouter, ", HashTable size :", TABLE_SIZE);
            std::vector<int> rowouter_cols;
            
            //Piro::logger::warning("Hashkeys:", Hash_keys_V);
            //Piro::logger::warning("Hashvalues:", Hash_val_V);
            // std::cout << Hash_val_V[141] << ", "<< Hash_keys_V[141] <<std::endl;
            // extract rowouter
            for(int co = rowouter; co < N; co++){
                float valofrow = Piro::methods::lookup(rowouter, co, N, Hash_keys_V, Hash_val_V, TABLE_SIZE);
                if(std::abs(valofrow) > 1E-6){
                    rowouter_cols.push_back(co);
                }
            }
            Piro::logger::warning("row outer", rowouter_cols);
            
            for (int r = rowouter + 1; r < N; ++r) {
                if(Piro::methods::lookup(r, rowouter, N, Hash_keys_V, Hash_val_V, TABLE_SIZE) == 0.0){
                    continue;
                }

                for (int col : rowouter_cols) {
                    // std::cout << col << ", ";
                    // if(col < rowouter) continue;
                    Piro::methods::lookupandset(r, col, N, 0.0f, Hash_keys_V, Hash_val_V, TABLE_SIZE);
                    /*
                    if(lookup(r, col, N, Hash_keys_V, Hash_val_V, TABLE_SIZE) == 0.0){
                        // set the col in the hash table directly
                        ind = r * N + col;
                        // std::cout << "ind = " << ind << std::endl;
                        sethash(ind, 0.0f, TABLE_SIZE, Hash_keys_V, Hash_val_V);
                        // std::cout << col << ", ";
                        
                    }*/
                }

            }
            
            Piro::logger::debug("Inserting 0s finished");
            //Piro::logger::warning("Hashkeys:", Hash_keys_V);
            //Piro::logger::warning("Hashvalues:", Hash_val_V);
            // query(83, Hash_keys_V, Hash_val_V, TABLE_SIZE);
            //hash_to_dense_and_print(Hash_keys_V, Hash_val_V, N, TABLE_SIZE);
            //Piro::logger::warning("Hashkeys:", Hash_keys_V);
            //Piro::logger::warning("Hashvalues:", Hash_val_V);
            err = clEnqueueWriteBuffer(queue, LFkeys.buffer, CL_FALSE, 
                                        0, 
                                        sizeof(int) * TABLE_SIZE,
                                        Hash_keys_V.data(), 
                                        0, nullptr, &event0);
            // assert(Lap_rowptr_V.data() + rowouter != nullptr);

            err = clEnqueueWriteBuffer(queue, LFvalues.buffer, CL_FALSE, 
                                        0, 
                                        sizeof(float) * TABLE_SIZE,
                                        Hash_val_V.data(), 
                                        0, nullptr, &event1);

            // Wait for all transfers to complete
            clWaitForEvents(2, events);
            Piro::logger::debug("Map memory object finished");
            // std::cout << cd.rowpointers[N] << std::endl;
            limit = (N - rowouter - 1) * (N - rowouter);
            // int mws = static_cast<int>(maxWorkGroupSize);
            size_t nnz = (size_t)limit;
            size_t local = (size_t)maxWorkGroupSize; // or whatever max workgroup size your device supports
            // Piro::logger::debug("nnz :", nnz, "local :", mws);
            globalWorkSize[0] = ((nnz + local - 1) / local) * local;
            localWorkSize[0] = local;
            // size_t localWorkSize[1] = { globalWorkSize[0] / 4 };
            err |= clSetKernelArg(kernelfilterarray, 5, sizeof(cl_int), &limit);
            err |= clSetKernelArg(kernelfilterarray, 3, sizeof(cl_int), &rowouter);
            err = clEnqueueNDRangeKernel(queue, kernelfilterarray, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
            clFinish(queue);
            Piro::logger::debug("Kernel finished");
            Piro::opencl_utilities::copyCL_offset<float>(queue, LFvalues.buffer, Hash_val_V, 0, TABLE_SIZE, &event4);
            Piro::opencl_utilities::copyCL_offset<int>(queue, LFkeys.buffer, Hash_keys_V, 0, TABLE_SIZE, &event5);
            Piro::logger::debug("CopyCL");
            Piro::logger::debug("Erased");
            //hash_to_dense_and_print(Hash_keys_V, Hash_val_V, N, TABLE_SIZE);
            // query(83, Hash_keys_V, Hash_val_V, TABLE_SIZE);
            
        }
    }
}
