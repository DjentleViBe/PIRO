#ifndef openclutilities_hpp
#define openclutilities_hpp

#include <vector>
#include <CL/opencl.h>

namespace Piro{
    class kernels{
        public:
            static kernels& getInstance(){
                static kernels instance;
                return instance;
            }
            kernels(const kernels&) = delete;
            kernels& operator=(const kernels&) = delete;
            enum ParameterIndex{
                PROGRAM_MATH, KERNEL_MATH, PROGRAM, KERNEL, CONTEXT, QUEUE,
                MAXWORKGROUPSIZE, COMPILE
            };
            template<typename T>
            void setvalue(const ParameterIndex paramindex, const T& val) {
                parameters[paramindex] = val;
            }
            // Specialization for arrays (e.g., std::vector)
            template<typename T>
            void setvalue(const ParameterIndex paramindex, const std::vector<T>& val) {
                parameters[paramindex] = val;
            }
            template<typename T>
            T& getvalue(const ParameterIndex paramindex) {
                if (parameters.find(paramindex) == parameters.end()) {
                    std::cout << "bad variant : " << paramindex;
                    throw std::runtime_error("Parameter not set");
                }
                if (!std::holds_alternative<T>(parameters.at(paramindex))) {
                    std::cout << "bad variant : " << paramindex;
                    throw std::bad_variant_access(); // Throwing a specific exception for mismatched types
                }
            return std::get<T>(parameters.at(paramindex));
            }

        private:
            kernels() = default;
            using ParamValue = std::variant<int, std::vector<cl_kernel>, std::vector<cl_program>,
                                            cl_context, cl_command_queue, cl_uint, cl_program, cl_kernel>;
            std::unordered_map<ParameterIndex, ParamValue> parameters;   
    };   
}

namespace Piro::opencl_utilities{
    template <typename T>
    std::vector<T> copyCL(cl_command_queue queue, cl_mem memC, int N, cl_event *event6);

    template <typename U>
    int copyCL_offset(cl_command_queue queue, cl_mem memC, std::vector<U>& Lap, int offset, int N, cl_event *event6);
}

#endif