#ifndef process_hpp
#define process_hpp

#include <operatoroverload.hpp>
#include <datatypes.hpp>
#include <ctime>
#include <string>

namespace Piro{
    int solve();
    class math_operations{
        public:
            int idx(int i, int j, int k, int N_x, int N_y);
    };

    class process{
        private:
            // bool sgm = false;
        public:
            /**
             * @brief Match name to variable.
             *
             * Given a scalar / vector name, compute the correspinding index number form the 
             * cell data array.
             * @param var Scalar / vector name
             * @return index number
             */
            int matchscalartovar(std::string var);
            /**
             * @brief Match name to variable.
             *
             * Given a constant name, compute the correspinding index number form the 
             * cell data array.
             * @param var constant name
             * @return index number
             */
            int matchconstanttovar(std::string var);
            int matchvectortovar(std::string var);
            /**
             * @brief OpenCL function to calulate time derivate of a variable.
             *
             * Computes the matrix which represents the time 
             * derivative of the input scalar / vector.
             * @param var Scalar / vector name
             * @return CLBuffer location
             */
            CLBuffer ddt_r(std::string var);
            /**
             * @brief OpenCL function to calulate laplacian matrix of a scalar / vector variable.
             *
             * Computes the laplacian of the input scalar / vector and multiples by a constant.
             * @param var1 Scalar / vector name
             * @param var2  Constant name
             * @return CLBuffer vector <rowpointers, coluns, values> of the CSR matrix.
             */
            std::vector<CLBuffer> laplacian(std::string var1, std::string var2);

            std::vector<CLBuffer> gradient(std::string var1);

            std::vector<CLBuffer> vector(std::string var1);

            std::vector<CLBuffer> div(std::string var1, std::string var2);
    };

    class scalarMatrix{
        private:
            CLBuffer smatrix;
        public:
            scalarMatrix(CLBuffer SM);
            void Solve(float currenttime);
        };
        
}

#endif