#ifndef python_performance_hpp
#define python_performance_hpp
#include <vector>

int callpython();
std::vector<float> mul_using_numpy(std::vector<std::vector<float>> A, std::vector<float> B);
#endif