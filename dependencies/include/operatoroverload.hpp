// operator_overload.hpp

#ifndef OPERATOROVERLOADHPP
#define OPERATOROVERLOADHPP

#include <vector>

std::vector<float> operator+(const std::vector<float> &v1, const std::vector<float> &v2);
std::vector<std::vector<float>> operator+(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2);
std::vector<float> operator-(const std::vector<float> &v1, const std::vector<float> &v2);
std::vector<std::vector<float>> operator-(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2);
std::vector<float> operator*(const std::vector<float> &v1, const std::vector<float> &v2);
std::vector<std::vector<float>> operator*(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2);
std::vector<float> operator/(const std::vector<float> &v1, const std::vector<float> &v2);
std::vector<std::vector<float>> operator/(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2);
#endif // OPERATOR_OVERLOAD_HPP