// operator_overload.hpp

#ifndef OPERATOROVERLOADHPP
#define OPERATOROVERLOADHPP
#include <vector>

namespace Piro{
    class CLBuffer{
        public:
            cl_mem buffer;
            CLBuffer operator=(const std::vector<CLBuffer>& other);
    
    };
}

Piro::CLBuffer operator*(Piro::CLBuffer partA, Piro::CLBuffer partB);
Piro::CLBuffer operator*(const float partA, Piro::CLBuffer partB);
Piro::CLBuffer operator+(Piro::CLBuffer partA, Piro::CLBuffer partB);
Piro::CLBuffer operator+(const float partA, Piro::CLBuffer partB);
Piro::CLBuffer operator-(Piro::CLBuffer partA, Piro::CLBuffer partB);
Piro::CLBuffer operator-(const float partA, Piro::CLBuffer partB);
Piro::CLBuffer operator/(Piro::CLBuffer partA, Piro::CLBuffer partB);
Piro::CLBuffer operator/(const float partA, Piro::CLBuffer partB);

namespace Piro{
    class CellDataGPU{
        public:
            std::vector<Piro::CLBuffer> values_gpu;
            std::vector<Piro::CLBuffer> laplacian_csr;
            std::vector<Piro::CLBuffer> gradient;
    };
};

namespace Piro{
    class Equation{
        public:
            cl_mem operandvalues;
            cl_mem operandcolumns;
            cl_mem operandrowptr;
            cl_mem operandrows;
            int sparsecount;
    };
};

extern Piro::CellDataGPU CDGPU;
extern Piro::Equation RHS;
const float EPSILON = 1E-6;
#endif // OPERATOR_OVERLOAD_HPP