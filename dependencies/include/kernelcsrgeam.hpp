#ifndef kernelcsrgeam_hpp
#define kernelcsrgeam_hpp

#include <datatypes.hpp>
#include <vector>
#include <operatoroverload.hpp>

namespace Piro::kernelmethods{
    void csrgeam(std::vector<CLBuffer> partA, std::vector<CLBuffer> partB);
}
#endif