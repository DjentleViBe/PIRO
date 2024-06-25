#ifndef solve_hpp
#define solve_hpp

#include "datatypes.hpp"
#include "preprocess.hpp"
#include <iostream>

extern Giro::SolveParams SP;
int solve();
namespace Giro{
    class MathOperations{
        public:
            std::vector<float> addVectors(const std::vector<float> &v1, const std::vector<float> &v2) {
                if (v1.size() != v2.size()) {
                    throw std::invalid_argument("Vectors must be of the same size");
                }

                std::vector<float> result(v1.size());
                for (size_t i = 0; i < v1.size(); ++i) {
                    result[i] = v1[i] + v2[i];
                }

                return result;
            }

            std::vector<float> subtractVectors(const std::vector<float> &v1, const std::vector<float> &v2) {
                if (v1.size() != v2.size()) {
                    throw std::invalid_argument("Vectors must be of the same size");
                }

                std::vector<float> result(v1.size());
                for (size_t i = 0; i < v1.size(); ++i) {
                    result[i] = v1[i] + v2[i];
                }

                return result;
            }

            std::vector<float> multiplyVectors(const std::vector<float> &v1, const std::vector<float> &v2) {
                if (v1.size() != v2.size()) {
                    throw std::invalid_argument("Vectors must be of the same size");
                }

                std::vector<float> result(v1.size());
                for (size_t i = 0; i < v1.size(); ++i) {
                    result[i] = v1[i] * v2[i];
                }

                return result;
            }

            std::vector<float> divideVectors(const std::vector<float> &v1, const std::vector<float> &v2) {
                if (v1.size() != v2.size()) {
                    throw std::invalid_argument("Vectors must be of the same size");
                }

                std::vector<float> result(v1.size());
                for (size_t i = 0; i < v1.size(); ++i) {
                    result[i] = v1[i] / v2[i];
                }

                return result;
            }

            // Function to add two 2D matrices
            std::vector<std::vector<float>> addMatrices(const std::vector<std::vector<float>>& matrix1, 
                                                        const std::vector<std::vector<float>>& matrix2) {
                // Check if matrices have the same dimensions
                if (matrix1.size() != matrix2.size() || matrix1[0].size() != matrix2[0].size()) {
                    throw std::invalid_argument("Matrices must have the same dimensions for addition.");
                }

                // Initialize result matrix with dimensions of matrix1
                std::vector<std::vector<float>> result(matrix1.size(), std::vector<float>(matrix1[0].size(), 0.0));

                // Perform matrix addition
                for (size_t i = 0; i < matrix1.size(); ++i) {
                    for (size_t j = 0; j < matrix1[0].size(); ++j) {
                        result[i][j] = matrix1[i][j] + matrix2[i][j];
                    }
                }
                return result;
            }

            std::vector<std::vector<float>> subtractMatrices(const std::vector<std::vector<float>>& matrix1, 
                                                        const std::vector<std::vector<float>>& matrix2) {
                // Check if matrices have the same dimensions
                if (matrix1.size() != matrix2.size() || matrix1[0].size() != matrix2[0].size()) {
                    throw std::invalid_argument("Matrices must have the same dimensions for addition.");
                }

                // Initialize result matrix with dimensions of matrix1
                std::vector<std::vector<float>> result(matrix1.size(), std::vector<float>(matrix1[0].size(), 0.0));

                // Perform matrix addition
                for (size_t i = 0; i < matrix1.size(); ++i) {
                    for (size_t j = 0; j < matrix1[0].size(); ++j) {
                        result[i][j] = matrix1[i][j] - matrix2[i][j];
                    }
                }
                return result;
            }

            std::vector<std::vector<float>> multiplyMatrices(const std::vector<std::vector<float>>& matrix1, 
                                                    const std::vector<std::vector<float>>& matrix2) {
                // Check if matrices have the same dimensions
                if (matrix1.size() != matrix2.size() || matrix1[0].size() != matrix2[0].size()) {
                    throw std::invalid_argument("Matrices must have the same dimensions for addition.");
                }

                // Initialize result matrix with dimensions of matrix1
                std::vector<std::vector<float>> result(matrix1.size(), std::vector<float>(matrix1[0].size(), 0.0));

                // Perform matrix addition
                for (size_t i = 0; i < matrix1.size(); ++i) {
                    for (size_t j = 0; j < matrix1[0].size(); ++j) {
                        result[i][j] = matrix1[i][j] * matrix2[i][j];
                    }
                }
                return result;
            }

            std::vector<std::vector<float>> divideMatrices(const std::vector<std::vector<float>>& matrix1, 
                                                        const std::vector<std::vector<float>>& matrix2) {
                // Check if matrices have the same dimensions
                if (matrix1.size() != matrix2.size() || matrix1[0].size() != matrix2[0].size()) {
                    throw std::invalid_argument("Matrices must have the same dimensions for addition.");
                }

                // Initialize result matrix with dimensions of matrix1
                std::vector<std::vector<float>> result(matrix1.size(), std::vector<float>(matrix1[0].size(), 0.0));

                // Perform matrix addition
                for (size_t i = 0; i < matrix1.size(); ++i) {
                    for (size_t j = 0; j < matrix1[0].size(); ++j) {
                        result[i][j] = matrix1[i][j] / matrix2[i][j];
                    }
                }
            return result;
            }

            std::vector<std::vector<float>> dotMatrices2(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
            int rows_A = A.size();
            int cols_A = A[0].size();
            int cols_B = B[0].size();

            // Initialize the result matrix with dimensions rows_A x cols_B
            std::vector<std::vector<float>> result(rows_A, std::vector<float>(cols_B, 0));

            // Perform matrix multiplication
            for (int i = 0; i < rows_A; ++i) {
                for (int j = 0; j < cols_B; ++j) {
                    for (int k = 0; k < cols_A; ++k) {
                        result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }

            return result;
        }

        std::vector<float> dotMatrices(const std::vector<std::vector<float>>& A, const std::vector<float>& B) {
            int m = A.size();    // Number of rows in A
            int n = A[0].size(); // Number of columns in A (should be equal to size of B)

            // Resulting vector C will have size m
            std::vector<float> C(m, 0.0);

            // Perform matrix-vector multiplication
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    C[i] += A[i][j] * B[j];
                }
            }

            return C;
        }
    };

    class Solve{
        public:

            int matchscalartovar(std::string var){
                    for(int v = 0; v < MP.AMR[0].CD.size(); v++){
                        if(var == MP.AMR[0].CD[v].Scalars){
                            return v;
                        }
                    }
                    return 0;
                }

            std::vector<std::vector<float>> ddt(std::string var){
                int ind = matchscalartovar(var);
                int n = std::cbrt(MP.AMR[0].CD[ind].values.size());
                std::vector<std::vector<float>> A(n*n*n, std::vector<float>(n*n*n, 0.0));
                for (int i = 0; i < n*n*n; ++i) {
                    A[i][i] = MP.AMR[0].CD[ind].values[i];  // 1.0 or any other desired value
                }
                return A;
            }

            std::vector<float> ddt_r(std::string var){
                int ind = matchscalartovar(var);
                int n = std::cbrt(MP.AMR[0].CD[ind].values.size());
                
                std::vector<float>A (n*n*n, 0.0);
                for (int i = 0; i < n*n*n; i++) {
                    A[i] = MP.AMR[0].CD[ind].values[i];  // 1.0 or any other desired value
                    //std::cout << A[i] << " ";
                }
                return A;
            }

            std::vector<std::vector<float>> ddc(std::string var, float value){
                int ind = matchscalartovar(var);
                int n = std::cbrt(MP.AMR[0].CD[ind].values.size());
                std::vector<std::vector<float>> A(n*n*n, std::vector<float>(n*n*n, 0.0));
                for (int i = 0; i < n*n*n; ++i) {
                    for (int j = 0; j < n*n*n; ++j) {
                        A[i][j] = value;  // 1.0 or any other desired value
                    }   
                }   
                return A;
            }

            std::vector<float> ddc_r(std::string var, float value){
                int ind = matchscalartovar(var);
                int n = std::cbrt(MP.AMR[0].CD[ind].values.size());
                std::vector<float>A(n*n*n, 0.0);
                for (int i = 0; i < n*n*n; ++i) {
                        A[i] = value;  // 1.0 or any other desired value
                }   
                return A;
            }

            std::vector<std::vector<float>> laplacian(std::string var){
                int ind = matchscalartovar(var);
                int n = std::cbrt(MP.AMR[0].CD[ind].values.size());
                // matrix ensemble
                // Initialize a 2D vector (matrix) of size n x n with zeros
                std::vector<std::vector<float>> A(n*n*n, std::vector<float>(n*n*n, 0.0));
                float subd = SP.deltaT / (SP.delta[0] * SP.delta[0]);
                float supd = SP.deltaT / (SP.delta[0] * SP.delta[0]);
                float d = -6.0 * SP.deltaT / (SP.delta[0] * SP.delta[0]);
                // Set the main diagonal (index 0)
                for (int i = 0; i < n*n*n; ++i) {
                    A[i][i] = d;  // 1.0 or any other desired value
                }

                // Set the subdiagonal (index -1)
                for (int i = 1; i < n*n*n; ++i) {
                    A[i][i - 1] = subd;  // -1.0 or any other desired value
                }

                // Set the superdiagonal (index +1)
                for (int i = 0; i < n*n*n - 1; ++i) {
                    A[i][i + 1] = supd;  // -1.0 or any other desired value
                }

                return A;
            }

            std::vector<float> laplacian_r(std::string var){
                int ind = matchscalartovar(var);
                int n = std::cbrt(MP.AMR[0].CD[ind].values.size());
                std::vector<float> prop = MP.AMR[0].CD[ind].values;
                // matrix ensemble
                // Initialize a 2D vector (matrix) of size n x n with zeros
                std::vector<std::vector<float>> A(n*n*n, std::vector<float>(n*n*n, 0.0));
                float subd = SP.deltaT / (SP.delta[0] * SP.delta[0]);
                float supd = SP.deltaT / (SP.delta[0] * SP.delta[0]);
                float d = -6.0 * SP.deltaT / (SP.delta[0] * SP.delta[0]);
                // Set the main diagonal (index 0)
                for (int i = 0; i < n*n*n; ++i) {
                    A[i][i] = d;  // 1.0 or any other desired value
                }

                // Set the subdiagonal (index -1)
                for (int i = 1; i < n*n*n; ++i) {
                    A[i][i - 1] = subd;  // -1.0 or any other desired value
                }

                // Set the superdiagonal (index +1)
                for (int i = 0; i < n*n*n - 1; ++i) {
                    A[i][i + 1] = supd;  // -1.0 or any other desired value
                }
                MathOperations dM;
                return dM.dotMatrices(A, prop);
            }

        };

        
};

#endif