#ifndef bc_hpp
#define bc_hpp

#include <vector>
#include <CL/opencl.h>
#include <unordered_map>
#include <variant>
#include <iostream>

namespace Piro::bc{
    /**
     * @brief Prepare Boundary Condition.
     *
     * This function populates the cell indices that needs to be exported.
     *
     */
    void prepbc();
    /**
     * @brief Initialise Boundary Condition.
     *
     * This function initialises Boundary Condition based on BC_PROPERTY,
     * BC_TYPE and BC_VALUE
     */
    void initbc();
    /**
     * @brief Read Boundary Condition.
     *
     * This function reads the paramters for the Boundary Condiion from
     * the .ini file.
     */
    void readbc();
    /**
     * @brief Set Boundary Condition.
     *
     * This function sets the values for the cells belonging to
     * the boundary.
     */
    void setbc();
    /**
     * @brief Set Boundary Condition - OpenCL function.
     * @param ind Boundary number 
     * This function calls a an OpenCL kernel to set the values assigned 
     * to the boundary cells as per on host.
     */
    void opencl_setBC(int ind);
    /**
     * @brief Init Boundary Condition. - OpenCL function
     *
     * This function calls a kernel that initialises the cells belonging 
     * to the boundary.
     */
    void opencl_initBC();
    /**
     * @brief Set Boundary Condition - OpenCL function.
     * @param ind Boundary number 
     * This function calls a kernel that sets the values of the cells 
     * belonging to the boundary.
     */
    void opencl_setBC(int ind);
    /**
     * @brief Check if the point is on boundary.
     * @param x x index 
     * @param y y index 
     * @param z z index 
     * This function compues whether a given point lies on the boundary of the domain
     * @return TRUE if cell exists on boundary.
     */
    bool isBoundaryPoint(int x, int y, int z);
    /**
     * @brief [Cell indices to print.]
     *
     * [Boundary values and cell indices to export.]
     */
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