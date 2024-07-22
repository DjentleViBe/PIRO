#ifndef solve_hpp
#define solve_hpp

#include "datatypes.hpp"
#include "preprocess.hpp"
#include "bc.hpp"
#include "extras.hpp"
#include "postprocess.hpp"
#include <iostream>
#include <immintrin.h>
#include <ctime>
#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#endif
#include "gpuinit.hpp"

extern Giro::SolveParams SP;
extern char* dt;
extern std::time_t now;

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
            std::cout << "matmulstarted" << std::endl;
            print_time();
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
            std::cout << "matmulend" << std::endl;
            print_time();
            return C;
        }

        std::vector<float> dotMatricesSIMD(const std::vector<std::vector<float>>& A, const std::vector<float>& B) {
            std::cout << "matmulstarted" << std::endl;
            print_time();
            int m = A.size();    // Number of rows in A
            int n = A[0].size(); // Number of columns in A (should be equal to size of B)

            // Ensure B's size matches A's column count
            if (B.size() != n) {
                throw std::invalid_argument("The size of vector B must match the number of columns in matrix A.");
            }

            // Resulting vector C will have size m
            std::vector<float> C(m, 0.0);

            // Perform matrix-vector multiplication with AVX
            for (int i = 0; i < m; ++i) {
                const float* row = A[i].data();
                __m256 sum = _mm256_setzero_ps(); // Initialize sum vector to zero

                int j = 0;
                for (; j <= n - 8; j += 8) {
                    __m256 a = _mm256_loadu_ps(&row[j]);
                    __m256 b = _mm256_loadu_ps(&B[j]);
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
                }

                // Reduce sum vector to a single value
                __m128 sum_low = _mm256_castps256_ps128(sum);
                __m128 sum_high = _mm256_extractf128_ps(sum, 1);
                sum_low = _mm_add_ps(sum_low, sum_high);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                float partial_sum = _mm_cvtss_f32(sum_low);

                // Process any remaining elements
                for (; j < n; ++j) {
                    partial_sum += row[j] * B[j];
                }
                C[i] = partial_sum;
            }
            std::cout << "matmulend" << std::endl;
            print_time();
            return C;
        }

        /*std::vector<float> dotMatricesOpenCL(float* A_ptr, std::vector<float>& B) {
            
            uint m = MP.n[0] * MP.n[1] * MP.n[2];     // Number of rows in A
            uint n = m; // Number of columns in A (should be equal to size of B)
            uint k = 1;
            // Ensure B's size matches A's column count
            if (B.size() != m) {
                throw std::invalid_argument("The size of vector B must match the number of columns in matrix A.");
            }

            float* B_ptr = B.data();
            
            opencl_call(A_ptr, B_ptr, 10, m, n, k);
            
            return B;
        }*/

        std::vector<float> laplacianOpenCL(std::vector<float>& B) {
            
            uint m = MP.n[0] * MP.n[1] * MP.n[2];     // Number of rows in A
            uint n = m; // Number of columns in A (should be equal to size of B)
            uint k = 1;
            // Ensure B's size matches A's column count
            if (B.size() != m) {
                throw std::invalid_argument("The size of vector B must match the number of columns in matrix A.");
            }
            float* B_ptr = B.data();
            opencl_laplacian(B_ptr, 10, m, n, k);
            
            return B;
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

        int idx(int i, int j, int k, int N_x, int N_y) {
            return i + j * N_x + k * N_x * N_y;
        }

        std::vector<std::vector<float>> generatevectormatrix(std::vector<float> U){
            std::vector<std::vector<float>> result(MP.n[0]*MP.n[1]*MP.n[2], std::vector<float>(MP.n[0]*MP.n[1]*MP.n[2], 0));
            for (int k = 0; k < MP.n[2]; ++k) {
                for (int j = 0; j < MP.n[1]; ++j) {
                    for (int i = 0; i < MP.n[0]; ++i) {
                        int l = idx(i, j, k, MP.n[0], MP.n[1]);

                        // Diagonal entry
                        result[l][l] = 0;

                        // Off-diagonal entries
                        if (i > 0) {
                            result[l][idx(i-1, j, k, MP.n[0], MP.n[1])] = U[idx(i-1, j, k, MP.n[0], MP.n[1])];
                        }
                        if (i < MP.n[0] - 1) {
                            result[l][idx(i+1, j, k, MP.n[0], MP.n[1])] = U[idx(i+1, j, k, MP.n[0], MP.n[1])];
                        }
                        if (j > 0) {
                            result[l][idx(i, j-1, k, MP.n[0], MP.n[1])] = U[MP.n[0]*MP.n[1]*MP.n[2] + idx(i, j-1, k, MP.n[0], MP.n[1])];
                        }
                        if (j < MP.n[1] - 1) {
                            result[l][idx(i, j+1, k, MP.n[0], MP.n[1])] = U[MP.n[0]*MP.n[1]*MP.n[2] + idx(i, j+1, k, MP.n[0], MP.n[1])];
                        }
                        if (k > 0) {
                            result[l][idx(i, j, k-1, MP.n[0], MP.n[1])] = U[2 * MP.n[0]*MP.n[1]*MP.n[2] + idx(i, j, k-1, MP.n[0], MP.n[1])];
                        }
                        if (k < MP.n[2] - 1) {
                            result[l][idx(i, j, k+1, MP.n[0], MP.n[1])] = U[2 * MP.n[0]*MP.n[1]*MP.n[2] + idx(i, j, k+1, MP.n[0], MP.n[1])];
                        }
                    }
                }
            }
            return result;
        }
    };

    class Solve{
        private:
            bool sgm = false;
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

                //return mul_using_numpy(scalapmatrix, prop);
                return dM.laplacianOpenCL(prop);
            }

            std::vector<float> grad_r(std::string var1, std::string var2){
                int ind1 = matchscalartovar(var1);
                int ind2 = matchscalartovar(var2);
                MathOperations dM;
                if(sgm == false){
                    scagradmatrix = dM.multiplyMatrices(scagradmatrix, dM.generatevectormatrix(MP.AMR[0].CD[ind2].values));
                    sgm = true;
                }
                //printMatrix(scagradmatrix);
                if(MP.AMR[0].CD[ind1].type == 0){
                    return dM.dotMatrices(scagradmatrix,  MP.AMR[0].CD[ind1].values);
                }else{
                    return MP.AMR[0].CD[ind1].values;
                }
                
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
                    
                    std::cout << "Post processing started" << std::endl;
                    print_time();
                    postprocess("T");
                    print_time();
                    std::cout << "Post processing finished" << std::endl;
                }
        };
        
};

#endif