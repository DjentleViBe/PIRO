#include <cstdint>
#include <vector>
#include <cmath>

namespace Piro::math_operations{
    bool is_prime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        for (int i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0)
                return false;
        }
        return true;
    }

    int next_prime(int n) {
        while (!is_prime(n)) ++n;
        return n;
    }

    uint64_t nextPowerOf2(uint64_t N) {
        if (N <= 1) return 1;
        N--;
        N |= N >> 1;
        N |= N >> 2;
        N |= N >> 4;
        N |= N >> 8;
        N |= N >> 16;
        N |= N >> 32;  // Safe for 64-bit numbers
        return N + 1;
    }

    int idx(int i, int j, int k, int N_x, int N_y) {
        return i + j * N_x + k * N_x * N_y;
    }

    int index(int i, int j, int k, int Nx, int Ny) {
        return i + j * Nx + k * Nx * Ny;
    }

    float calc_length_3d(std::vector<float> coord, std::vector<float> center){
        return pow((pow((coord[0] - center[0]), 2) 
                    + pow((coord[1] - center[1]), 2) 
                    + pow((coord[2] - center[2]), 2)), 0.5);
    }
    
}