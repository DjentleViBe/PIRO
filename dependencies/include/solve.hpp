#ifndef solve_hpp
#define solve_hpp

#include "datatypes.hpp"
#include "preprocess.hpp"
#include <iostream>

extern Giro::SolveParams SP;
int solve();
namespace Giro{
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
                std::vector<std::vector<float>> A(n, std::vector<float>(n, 0.0));
                for (int i = 0; i < n; ++i) {
                    A[i][i] = MP.AMR[0].CD[ind].values[n * i];  // 1.0 or any other desired value
                }
                return A;
            }

            std::vector<std::vector<float>> ddc(std::string var, float value){
                int ind = matchscalartovar(var);
                int n = std::cbrt(MP.AMR[0].CD[ind].values.size());
                std::vector<std::vector<float>> A(n, std::vector<float>(n, 0.0));
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        A[i][j] = value;  // 1.0 or any other desired value
                    }   
                }   
                return A;
            }

            std::vector<std::vector<float>> laplacian(std::string var){
                int ind = matchscalartovar(var);
                int n = std::cbrt(MP.AMR[0].CD[ind].values.size());
                // matrix ensemble
                // Initialize a 2D vector (matrix) of size n x n with zeros
                std::vector<std::vector<float>> A(n, std::vector<float>(n, 0.0));
                float subd = SP.deltaT / (SP.delta[0] * SP.delta[0]);
                float supd = SP.deltaT / (SP.delta[0] * SP.delta[0]);
                float d = -6.0 * SP.deltaT / (SP.delta[0] * SP.delta[0]);
                // Set the main diagonal (index 0)
                for (int i = 0; i < n; ++i) {
                    A[i][i] = d;  // 1.0 or any other desired value
                }

                // Set the subdiagonal (index -1)
                for (int i = 1; i < n; ++i) {
                    A[i][i - 1] = subd;  // -1.0 or any other desired value
                }

                // Set the superdiagonal (index +1)
                for (int i = 0; i < n - 1; ++i) {
                    A[i][i + 1] = supd;  // -1.0 or any other desired value
                }

                return A;
            }

        };
};

#endif