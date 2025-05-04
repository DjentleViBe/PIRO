#ifndef mathoperations_hpp
#define mathoperations_hpp
#include <cstdint>

namespace Piro::math_operations{
    bool is_prime(int n);
    int next_prime(int n);
    uint64_t nextPowerOf2(uint64_t N);
}
#endif