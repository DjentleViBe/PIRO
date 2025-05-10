#ifndef preprocess_hpp
#define preprocess_hpp

#include <datatypes.hpp>
#include <vector>
#include <string>
#include <CL/opencl.h>

namespace Piro{
    int laplacian_CSR_init();
    int preprocess(const std::string& name);
    void print_device_info(cl_device_id device);
    cl_int opencl_BuildProgram(cl_program program);
    cl_program opencl_CreateProgram(const char* dialog);
    int opencl_init();
    int opencl_build();
    int opencl_run();
    std::vector<unsigned char> readBinaryFile(const std::string& filePath, size_t& binarySize);
    int opencl_cleanup();
    int loadKernelsFromFile(const std::string& filename, 
        std::vector<std::string>& knm, 
        std::vector<std::string>& ksm);
    std::string readFile(const std::string& kernelName);
    std::vector<unsigned char> readBinaryFile(const std::string& filename);
    class INIT{
        public:
            static INIT& getInstance() {
                static INIT instance;
                return instance;
            }
            INIT(const INIT&) = delete;
            INIT& operator=(const INIT&) = delete;
            bool RHS_INIT;
            bool LAP_INIT;
            int ts;
        private:
            INIT() 
                : RHS_INIT(false), LAP_INIT(false), ts(0) {}
    };
}

#endif