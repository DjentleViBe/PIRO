#include <vector>
#include "../dependencies/include/solve.hpp"
#include "../dependencies/include/operatoroverload.hpp"
// Overloading the + operator for std::vector<float>
std::vector<float> operator+(const std::vector<float> &v1, const std::vector<float> &v2) {
    Giro::MathOperations add;
    return add.addVectors(v1, v2);
}

std::vector<std::vector<float>> operator+(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2) {
    Giro::MathOperations addM;
    return addM.addMatrices(v1, v2);
}

std::vector<float> operator-(const std::vector<float> &v1, const std::vector<float> &v2) {
    Giro::MathOperations sub;
    return sub.subtractVectors(v1, v2);
}

std::vector<std::vector<float>> operator-(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2) {
    Giro::MathOperations subM;
    return subM.subtractMatrices(v1, v2);
}

std::vector<float> operator*(const std::vector<float> &v1, const std::vector<float> &v2) {
    Giro::MathOperations mul;
    return mul.multiplyVectors(v1, v2);
}

std::vector<std::vector<float>> operator*(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2) {
    Giro::MathOperations mulM;
    return mulM.multiplyMatrices(v1, v2);
}

std::vector<float> operator/(const std::vector<float> &v1, const std::vector<float> &v2) {
    Giro::MathOperations div;
    return div.divideVectors(v1, v2);
}

std::vector<std::vector<float>> operator/(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2) {
    Giro::MathOperations divM;
    return divM.divideMatrices(v1, v2);
}
