#include <openclutilities.hpp>
#include <kernelprocess.hpp>
#include <CL/opencl.h>
#include <datatypes.hpp>
#include <logger.hpp>

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
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 8, sizeof(cl_mem), &partC.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 9, sizeof(cl_int), &n[0]);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 10, sizeof(cl_int), &n[1]);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 11, sizeof(cl_int), &N);
    clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL)[8], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
        
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 0, sizeof(cl_mem), &partD.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 1, sizeof(cl_mem), &LHS_vec);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 2, sizeof(cl_mem), &partC.buffer);
    clSetKernelArg(kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 3, sizeof(cl_uint), &N);
    clEnqueueNDRangeKernel(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE), kernels.getvalue<std::vector<cl_kernel>>(Piro::kernels::KERNEL_MATH)[0], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(kernels.getvalue<cl_command_queue>(Piro::kernels::QUEUE));
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
