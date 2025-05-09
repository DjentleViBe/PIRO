// operator_overload.hpp

#ifndef OPERATOROVERLOADHPP
#define OPERATOROVERLOADHPP
#include <vector>
#include <CL/opencl.h>

namespace Piro{
    class CLBuffer{
        public:
            cl_mem buffer;
            CLBuffer operator=(const std::vector<CLBuffer>& other);
    
    };
    CLBuffer operator*(CLBuffer partA, CLBuffer partB);
    CLBuffer operator*(const float partA, CLBuffer partB);
    CLBuffer operator+(CLBuffer partA, Piro::CLBuffer partB);
    CLBuffer operator+(const float partA, CLBuffer partB);
    CLBuffer operator-(CLBuffer partA, CLBuffer partB);
    CLBuffer operator-(const float partA, CLBuffer partB);
    CLBuffer operator/(CLBuffer partA, CLBuffer partB);
    CLBuffer operator/(const float partA, CLBuffer partB);
}

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

extern Piro::Equation RHS;
const float EPSILON = 1E-6;
#endif // OPERATOR_OVERLOAD_HPP