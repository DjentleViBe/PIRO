#include <openclutilities.hpp>
#include <kernelprocess.hpp>
#include <CL/opencl.h>
#include <datatypes.hpp>
#include <logger.hpp>
#include <extras.hpp>
#include <mathoperations.hpp>

void Piro::kernelsolve::CSR::TIMESCHEME_11(const std::vector<CLBuffer>& other, int N, int P, 
                                        std::vector<uint> n, float timestep, 
                                        Piro::CLBuffer partC, Piro::CLBuffer partD, 
                                        cl_mem RHS_vec, cl_mem LHS_vec){
                                        
    Piro::kernels& kernels = Piro::kernels::getInstance();
        
    size_t globalWorkSize[1] = { (size_t)N };
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 0, sizeof(cl_int), &N);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 1, sizeof(cl_int), &N);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 2, sizeof(cl_int), &P);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 3, sizeof(cl_mem), &other[2].buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 4, sizeof(cl_mem), &other[1].buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 5, sizeof(cl_mem), &other[0].buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 6, sizeof(cl_float), &timestep);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 7, sizeof(cl_mem), &RHS_vec);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 8, sizeof(cl_mem), &partD.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 9, sizeof(cl_int), &n[0]);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 10, sizeof(cl_int), &n[1]);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 11, sizeof(cl_int), &N);
    clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
    /*
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 0, sizeof(cl_mem), &partD.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 1, sizeof(cl_mem), &LHS_vec);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 2, sizeof(cl_mem), &partC.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 3, sizeof(cl_uint), &N);
    clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
    */
}

void Piro::kernelsolve::DENSE::TIMESCHEME_11(const std::vector<CLBuffer>& other, int N, int P, float timestep, 
                                                cl_mem partB, Piro::CLBuffer partC,
                                                Piro::CLBuffer partD){
    // A * x
    Piro::kernels& kernels = Piro::kernels::getInstance();
    size_t globalSize[2] = { (size_t)N, (size_t)N };
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[14], 0, sizeof(cl_mem), &other[0].buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[14], 1, sizeof(cl_mem), &partB);
    // result
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[14], 2, sizeof(cl_mem), &partC.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[14], 3, sizeof(cl_uint), &N);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[14], 4, sizeof(cl_uint), &N);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[14], 5, sizeof(cl_uint), &P);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[14], 6, sizeof(cl_uint), &timestep);
    clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[14], 2, NULL, globalSize, NULL, 0, NULL, NULL);
    clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));

    size_t globalWorkSize[1] = { (size_t)N };
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 0, sizeof(cl_mem), &partD.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 1, sizeof(cl_mem), &partB);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 2, sizeof(cl_mem), &partC.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 3, sizeof(cl_uint), &N);
    clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
}

void Piro::kernelsolve::HT::TIMESCHEME_11(const std::vector<CLBuffer>& other, int N, int TABLE_SIZE, 
                                            cl_mem partB, Piro::CLBuffer partC, 
                                            Piro::CLBuffer partD, float timestep){
    Piro::kernels& kernels = Piro::kernels::getInstance();
    size_t globalWorkSize[1] = { (size_t)N };
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[15], 0, sizeof(cl_mem), &other[0].buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[15], 1, sizeof(cl_mem), &other[1].buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[15], 2, sizeof(cl_mem), &partB);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[15], 3, sizeof(cl_mem), &partC.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[15], 4, sizeof(cl_int), &N);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[15], 5, sizeof(cl_int), &TABLE_SIZE);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[15], 6, sizeof(cl_uint), &timestep);
    clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[15], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));

    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 0, sizeof(cl_mem), &partD.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 1, sizeof(cl_mem), &partB);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 2, sizeof(cl_mem), &partC.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 3, sizeof(cl_uint), &N);
    clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));


}

void Piro::kernelsolve::COO::TIMESCHEME_11(const std::vector<CLBuffer>& other, int N, int nnz, 
                                        std::vector<uint> n, float timestep, 
                                        Piro::CLBuffer partC, Piro::CLBuffer partD, 
                                        cl_mem partB){
    Piro::kernels& kernels = Piro::kernels::getInstance();
    size_t globalWorkSize[1] = { (size_t)N };
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 0, sizeof(cl_int), &N);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 1, sizeof(cl_int), &N);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 2, sizeof(cl_int), &nnz);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 3, sizeof(cl_mem), &other[2].buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 4, sizeof(cl_mem), &other[1].buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 5, sizeof(cl_mem), &other[0].buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 6, sizeof(cl_float), &timestep);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 7, sizeof(cl_mem), &partB);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 8, sizeof(cl_mem), &partC.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 9, sizeof(cl_int), &n[0]);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 10, sizeof(cl_int), &n[1]);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 11, sizeof(cl_int), &N);
    clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[16], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));

    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 0, sizeof(cl_mem), &partD.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 1, sizeof(cl_mem), &partB);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 2, sizeof(cl_mem), &partC.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 3, sizeof(cl_uint), &N);
    clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

}


void Piro::kernelsolve::CSR::TIMESCHEME_12_40(const std::vector<CLBuffer>& other, int N, int P, 
                                        std::vector<uint> n, float timestep, 
                                        Piro::CLBuffer partC, Piro::CLBuffer partD, 
                                        cl_mem RHS_vec, cl_mem residuals){
        
    // Step 1 : Take LHS as initial value
    // Step 2 : Compute residual
    // A * x
    Piro::kernels& kernels = Piro::kernels::getInstance();
    Piro::SolveParams& SP = Piro::SolveParams::getInstance();
    Piro::SolveParams::JLI omega = SP.getvalue<Piro::SolveParams::JLI>(Piro::SolveParams::JLI_PARAM);
    size_t globalWorkSize[1] = { (size_t)N };
    float residual = 0.0f;
    size_t local = (size_t)kernels.getvalue<cl_uint>(Piro::kernels::MAXWORKGROUPSIZE); // or whatever max workgroup size your device supports
    // size_t global = ((N + local - 1) / local) * local;
    // size_t max_groups = (global / local);
    size_t current_size = N;
    /*cl_mem residual_buffer = clCreateBuffer(kernels.getvalue<cl_context>(Piro::kernels::CONTEXT),
                                        CL_MEM_READ_WRITE,
                                        sizeof(float) * N,
                                        nullptr, nullptr);
    */
    cl_event event1, event2, event3;
    clEnqueueCopyBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), RHS_vec, partD.buffer, 0, 0, sizeof(float) * N, 0, NULL, &event1);
    clWaitForEvents(1, &event1);
    float patternf = 0.0f;
    float residuals_old_cpu = 0.0f;
    // float temp = 1.0;
    for(int inner = 0; inner < omega.maxiter; inner++){
        // Piro::logger::info("y_k");
        // Piro::print_utilities::printCL(partD.buffer, N, 1);
        clEnqueueFillBuffer(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), partC.buffer, &patternf, sizeof(float), 0, N*sizeof(float), 0, NULL, &event3);
        clWaitForEvents(1, &event3);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 0, sizeof(cl_int), &N);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 1, sizeof(cl_int), &N);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 2, sizeof(cl_int), &P);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 3, sizeof(cl_mem), &other[2].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 4, sizeof(cl_mem), &other[1].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 5, sizeof(cl_mem), &other[0].buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 6, sizeof(cl_float), &timestep);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 7, sizeof(cl_mem), &partD.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 8, sizeof(cl_mem), &partC.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 9, sizeof(cl_int), &n[0]);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 10, sizeof(cl_int), &n[1]);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 11, sizeof(cl_int), &N);
        clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[25], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
        // partD.buffer = x^(k) (current iterate)
        // Piro::logger::info("M * x_k : ");
        // Piro::print_utilities::printCL(partC.buffer, N, 1);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[1], 0, sizeof(cl_mem), &partC.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[1], 1, sizeof(cl_mem), &RHS_vec);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[1], 2, sizeof(cl_mem), &partC.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[1], 3, sizeof(cl_uint), &N);
        clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[1], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
        // Piro::logger::info("RHS_vec");
        // Piro::print_utilities::printCL(RHS_vec, N, 1);
        // Piro::logger::info("b - (M * x_k)");
        // Piro::print_utilities::printCL(partC.buffer, N, 1);
        
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[12], 0, sizeof(cl_mem), &partD.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[12], 1, sizeof(cl_mem), &partC.buffer);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[12], 2, sizeof(cl_float), &omega.URF);
        clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[12], 3, sizeof(cl_uint), &N);
        clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[12], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
        // Piro::print_utilities::printCL(partC.buffer, N, 1);
        // Piro::logger::info("x(k) + omega * (b - (M * x(k)))");
        // Piro::print_utilities::printCL(partD.buffer, N, 1);
        
        
        std::vector<float> res = Piro::opencl_utilities::copyCL<float>(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), 
                                                                partC.buffer, N, &event2);
        float residuals_cpu = sqrt(Piro::vector_operations::sum_numbers(res)) / std::sqrt(static_cast<float>(N));
        float residual = (residuals_cpu - residuals_old_cpu) * (residuals_cpu - residuals_old_cpu);
        Piro::logger::info("Iteration :",inner, "Residuals_CPU :", residual);
        if(residual < omega.tolerance){
            Piro::logger::info("Converged!");
            break;
        }
        residuals_old_cpu = residuals_cpu;

    }
    // clReleaseMemObject(residual_buffer);

}