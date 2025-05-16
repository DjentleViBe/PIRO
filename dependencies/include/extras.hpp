#ifndef extras_hpp
#define extras_hpp

#include <vector>
#include <iostream>

namespace Piro::vector_operations{
    /**
     * @brief Flatten vector.
     *
     * This function flattens a multi dimensional vector to a single 
     * dimensional vector
     * @param twoDVector 2D vector.
     * @return Flattened vector.
     */
    std::vector<int> flattenvector(std::vector<std::vector<int>> twoDVector);
    std::vector<float> flattenvector(std::vector<std::vector<float>> twoDVector);
}
#endif