#ifndef printutilities_hpp
#define printutilities_hpp

namespace Piro::PrintUtilities{
    void printMatrix(const std::vector<std::vector<float>>& matrix);
    void printMatrix(const std::vector<std::vector<int>>& matrix);
    void printVector(const std::vector<float>& vec);
    void printVector(const std::vector<int>& vec);
    void printArray(float* array, uint size);
    void printCL(cl_mem memC, int N, int type);
    void printCLArray(cl_mem memC, int N, int type);
    void csr_to_dense_and_print(const std::vector<int>& row_pointer,
        const std::vector<int>& columns,
        const std::vector<float>& values,
        int N);
    void hash_to_dense_and_print(std::vector<int> Hashkeys, std::vector<float> HashValues, int N, int TABLE_SIZE);
    void print_time();
}
#endif