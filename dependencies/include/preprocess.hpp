#ifndef preprocess_hpp
#define preprocess_hpp

#include <datatypes.hpp>
#include <vector>
#include <string>
#include <CL/opencl.h>

namespace Piro{
    /**
     * @brief Preprocessing step of the simulation.
     *
     * This function read parameters set in the setup file.
     * @param name steup file name with filepath appended
     *
     * @return 0 upon SUCCESS.
     */
    int preprocess(const std::string& name);
    /**
     * @brief CPU / GPU device info.
     * @param device device ID
     *
     * This function prints device info geenrated by OpenCL libraries.
     *
     */
    void print_device_info(cl_device_id device);
    /**
     * @brief Program build.
     * @param program Program name
     *
     * This function builds program for OpenCL.
     *
     * @return 0 upon SUCCESS.
     */
    cl_int opencl_BuildProgram(cl_program program);
    /**
     * @brief program creation.
     * @param dialog An array of count pointers to optionally null-terminated character strings that make up the source code. 
     *
     * This function creates a program for OpenCL.
     *
     * @return 0 upon SUCCESS.
     */
    cl_program opencl_CreateProgram(const char* dialog);
    /**
     * @brief OpenCL initialisation.
     *
     * This function does initialisations required for OpenCL launch.
     *
     * @return 0 upon SUCCESS.
     */
    int opencl_init();
    /**
     * @brief OpenCL Compile time commands.
     *
     * This function executes commands required during OpenCL building, *.cl file 
     * are output to the kernel folder 
     *
     * @return 0 upon SUCCESS.
     */
    int opencl_build();
    /**
     * @brief OpenCL Run time commands.
     *
     * This function executes commands required during OpenCL run, *.cl files
     * are rn from the kernel folder
     *
     * @return 0 upon SUCCESS.
     */
    int opencl_run();
    /**
     * @brief Read a binary program.
     *
     * This function reads a compiled program in binary format.
     * @param filePath filelocation
     * @param binarySize size of the binary file
     *
     * @return content of the binary file.
     */
    std::vector<unsigned char> readBinaryFile(const std::string& filePath, size_t&);
    std::vector<unsigned char> readBinaryFile(const std::string& filename);
    int opencl_cleanup();
    /**
     * @brief Load kernel from source.
     *
     * This function loads a pre-compiled kernel from source.
     * @param filename file location
     * @param knm vector containing kernel names
     * @param ksm vector containing kernel source
     *
     * @return 0 upon SUCCESS.
     */
    int loadKernelsFromFile(const std::string& filename, 
                            std::vector<std::string>& knm, 
                            std::vector<std::string>& ksm);
    /**
     * @brief Read a file.
     *
     * This function reads a precompiled kernel file.
     *
     * @return contents of the file.
     */
    std::string readFile(const std::string& kernelName);
    
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