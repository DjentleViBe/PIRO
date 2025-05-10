#ifndef bc_hpp
#define bc_hpp

#include <vector>
#include <CL/opencl.h>
#include <unordered_map>
#include <variant>
#include <iostream>

namespace Piro::bc{
    void prepbc();
    void initbc();
    void readbc();
    void setbc();
    void opencl_setBC(int ind);
    void opencl_initBC();
    void opencl_setBC(int ind);
    class indices{  
        public:
            static indices& getInstance(){
                static indices instance;
                return instance;
            }

            indices(const indices&) = delete;
            indices& operator=(const indices&) = delete;
            enum ParameterIndex{
                // int
                IND, INDTOPRINT, INDTOPRINTVEC, BC_PROPERTY, BC_VALUE, BC_TYPE
            };
            template<typename T>
            void setvalue(const ParameterIndex paramindex, const T& val) {
                parameters[paramindex] = val;
            }

            template<typename T>
            T& getvalue(const ParameterIndex paramindex) {
                if (parameters.find(paramindex) == parameters.end()) {
                    std::cout << "bad variant (CellData - GPU) : " << paramindex;
                    throw std::runtime_error("Parameter not set");
                }
                return std::get<T>(parameters.at(paramindex));
            }
        private:
            indices() = default;
            using ParamValue = std::variant<int, std::vector<std::string>, std::vector<float>, std::vector<std::vector<int>>, std::vector<int>>;
            std::unordered_map<ParameterIndex, ParamValue> parameters;

    };   
}

#endif