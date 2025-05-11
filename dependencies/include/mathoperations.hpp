#ifndef mathoperations_hpp
#define mathoperations_hpp
#include <cstdint>
#include <vector>

namespace Piro::math_operations{
    /**
     * @brief Prime number.
     *
     * This function calculates whether a number is prime or not.
     *
     * @param n number to be tested. 
     * @return TRUE if prime.
     */
    bool is_prime(int n);
    /**
     * @brief Calcuate the next nearest prime number.
     *
     * This function return a number that is a prime number closest to
     * the input number.
     *
     * @param n inout number.
     * @return closest prime.
     */
    int next_prime(int n);
    /**
     * @brief next power of 2.
     *
     * This function calculates the nuext number that is a power of 2.
     *
     * @param N input number. 
     * @return next power of 2.
     */
    uint64_t nextPowerOf2(uint64_t N);
    /**
     * @brief idx function.
     *
     * This function calculates index given i, j, k indices.
     *
     * @param i index i. 
     * @param j index j.
     * @param k index k.
     * @param N_x Number of cells in X axis
     * @param N_y Number of cells in Y axis
     * @return index value.
     */
    int idx(int i, int j, int k, int N_x, int N_y);
    /**
     * @brief idx function.
     *
     * This function calculates index given i, j, k indices.
     *
     * @param i index i. 
     * @param j index j.
     * @param k index k.
     * @param Nx Number of cells in X axis
     * @param Ny Number of cells in Y axis
     * @return index value.
    */
    int index(int i, int j, int k, int Nx, int Ny);
    /**
     * @brief distance function.
     *
     * This function calculates the distance between 2 3D points.
     *
     * @param coord vector containing first coordinate. 
     * @param center vector containing second coordinate .
     * @return distance value.
    */
    float calc_length_3d(std::vector<float> coord, std::vector<float> center);
}
#endif