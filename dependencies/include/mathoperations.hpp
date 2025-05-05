#ifndef mathoperations_hpp
#define mathoperations_hpp
#include <cstdint>
#include <vector>

namespace Piro::math_operations{
    bool is_prime(int n);
    int next_prime(int n);
    uint64_t nextPowerOf2(uint64_t N);
    int idx(int i, int j, int k, int N_x, int N_y);
    int index(int i, int j, int k, int Nx, int Ny);
    float calc_length_3d(std::vector<float> coord, std::vector<float> center);
}
#endif