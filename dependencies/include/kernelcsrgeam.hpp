#ifndef kernelcsrgeam_hpp
#define kernelcsrgeam_hpp

#include <datatypes.hpp>
#include <vector>
#include <operatoroverload.hpp>

namespace Piro::kernelmethods{
    void csrgeam(std::vector<CLBuffer> partA, std::vector<CLBuffer> partB, int module);
    void csrgeam_2(std::vector<CLBuffer> partA, std::vector<CLBuffer> partB, int module);
}
#endif