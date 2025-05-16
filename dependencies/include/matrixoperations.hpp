#ifndef matrixoperations_hpp
#define matrixoperations_hpp

#include <operatoroverload.hpp>
#include <vector>

namespace Piro::matrix_operations{
    namespace HT{
    /**
     * @brief LU dcomposition function.
     *
     * This function prepares matrix for the LU decomposition method.
     *
     * @param other Open CL buffer to the RHS side of the equation. 
     */
    void lu_decomposition(const std::vector<CLBuffer>& other);
    }
}
#endif