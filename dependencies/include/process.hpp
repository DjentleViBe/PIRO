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
            int matchscalartovar(std::string var);
            int matchconstanttovar(std::string var);
            CLBuffer ddt_r(std::string var);
            const float ddc_r(std::string var);
            CLBuffer r(std::string var);
            CLBuffer laplacian_full(std::string var);
            std::vector<CLBuffer> laplacian_CSR(std::string var1, std::string var2);
            CLBuffer div_r(std::string var1, std::string var2);
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