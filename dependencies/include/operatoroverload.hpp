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

    std::vector<CLBuffer> operator&(std::vector<CLBuffer> partA, std::vector<CLBuffer> partB);
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
            static Equation& getInstance() {
                static Equation instance;
                return instance;
            }

            Equation(const Equation&) = delete;
            Equation& operator=(const Equation&) = delete;
            cl_mem operandvalues;
            cl_mem operandcolumns;
            cl_mem operandrowptr;
            cl_mem operandrows;
            int sparsecount;
        private:
            Equation() :
                sparsecount(0) {};
    };
};

#endif // OPERATOR_OVERLOAD_HPP