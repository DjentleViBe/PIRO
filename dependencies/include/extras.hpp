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
    /**
     * @brief Cumulative sum.
     *
     * This function calculates the cumulative sum of a array.
     * @param numbers 1D vector.
     * @return Sum.
     */
    float sum_numbers(const std::vector<float>& numbers);
    /**
     * @brief Subtract vectors.
     *
     * Subtracts second vector from first.
     * @param number1 1D vector.
     * @param number2 1D vector.
     * @return subtracted vector.
     */
    void subtractvectors(std::vector<float>& number1, const std::vector<float>& number2);
}
#endif