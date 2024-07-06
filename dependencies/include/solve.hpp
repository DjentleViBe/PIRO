#ifndef solve_hpp
#define solve_hpp

#include "datatypes.hpp"
#include "preprocess.hpp"
#include "bc.hpp"
#include "extras.hpp"
#include "postprocess.hpp"
#include <iostream>

extern Giro::SolveParams SP;

int solve();
namespace Giro{
    class MathOperations{
        public:
            std::vector<float> addVectors(const std::vector<float> &v1, const std::vector<float> &v2) {
                if (v1.size() != v2.size()) {
                    throw std::invalid_argument("Added vectors must be of the same size");
                }

                std::vector<float> result(v1.size());
                for (size_t i = 0; i < v1.size(); ++i) {
                    result[i] = v1[i] + v2[i];
                }

                return result;
            }

            std::vector<float> subtractVectors(const std::vector<float> &v1, const std::vector<float> &v2) {
                if (v1.size() != v2.size()) {
                    throw std::invalid_argument("Subtracted vectors must be of the same size");
                }

                std::vector<float> result(v1.size());
                for (size_t i = 0; i < v1.size(); ++i) {
                    result[i] = v1[i] + v2[i];
                }

                return result;
            }

            std::vector<float> multiplyVectors(const std::vector<float> &v1, const std::vector<float> &v2) {
                if (v1.size() != v2.size()) {
                    throw std::invalid_argument("Multiplied vectors must be of the same size");
                }

                std::vector<float> result(v1.size());
                for (size_t i = 0; i < v1.size(); ++i) {
                    result[i] = v1[i] * v2[i];
                }

                return result;
            }

            std::vector<float> divideVectors(const std::vector<float> &v1, const std::vector<float> &v2) {
                if (v1.size() != v2.size()) {
                    throw std::invalid_argument("Divided vectors must be of the same size");
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

        std::vector<std::vector<float>> convertTo6x3(std::vector<std::vector<float>> mtx){
            std::vector<std::vector<float>> result(mtx[0].size() * 3, std::vector<float>(mtx[0].size(), 0));
            for (int i = 0; i < mtx[0].size(); ++i) {
                for (int j = 0; j < mtx[0].size(); ++j) {
                    result[i][j] = mtx[i][j]; // Copying original elements
                    result[i + 3][j] = mtx[i][j]; // Copying again into the next set of rows
                }
            }
        return result;
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

            std::vector<float> ddt_r(std::string var){
                int ind = matchscalartovar(var);
                // int n = std::cbrt(MP.AMR[0].CD[ind].values.size());
                
                std::vector<float>A (MP.n[0] * MP.n[1] * MP.n[2], 0.0);
                for (int i = 0; i < MP.n[0] * MP.n[1] * MP.n[2]; i++) {
                    A[i] = MP.AMR[0].CD[ind].values[i];  // 1.0 or any other desired value
                    //std::cout << A[i] << " ";
                }
                return A;
            }

            std::vector<float> ddc_r(std::string var){
                int ind = matchscalartovar(var);
                // int n = std::cbrt(MP.AMR[0].CD[ind].values.size());
                std::vector<float>A(MP.n[0] * MP.n[1] * MP.n[2], 0.0);
                for (int i = 0; i < MP.n[0] * MP.n[1] * MP.n[2]; ++i) {
                        A[i] = MP.constantsvalues[ind];  // 1.0 or any other desired value
                }   
                return A;
            }

            std::vector<float> laplacian_r(std::string var){
                int ind = matchscalartovar(var);
                std::vector<float> prop = MP.AMR[0].CD[ind].values;
                // matrix ensemble
                // Initialize a 2D vector (matrix) of size n x n with zeros
                MathOperations dM;
                return dM.dotMatrices(scalapmatrix, prop);
            }

            std::vector<float> div_r(std::string var){
                int ind = matchscalartovar(var);
                std::vector<float> prop = MP.AMR[0].CD[ind].values;
                MathOperations dM;
                return dM.dotMatrices(scadivmatrix, prop);
            }

        };

        class scalarMatrix{
            private:
                std::vector<float> smatrix;
            public:
                scalarMatrix(std::vector<float> SM){
                    smatrix = SM;
                }

                void Solve(float currenttime){
                    ts = int(currenttime / SP.timestep);
                    std::cout << "Timestep : " << ts  << " / " << SP.totaltimesteps << std::endl;
                    // assign variables
                    MP.AMR[0].CD[0].values = smatrix;
                    // apply Boundary Conditions
                    setbc();
                    // export every timestep
                    postprocess("T");
                }
        };
        
};

#endif