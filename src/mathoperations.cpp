#include <cstdint>

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
}